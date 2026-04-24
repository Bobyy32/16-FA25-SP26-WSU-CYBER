#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training script for parallelized model training."""
__author__ = "parallel-training-team"

import logging as _logging_module
import os as _os_module
from collections.abc import Iterable as Ibltrble
from contextlib import nullcontext as NulCtxxt

import torch
import torch.distributed as distribs
import torch.optim as optims
import wandb as wdb
from datasets import load_dataset as dlsd
from torch.distributed.checkpoint.state_dict import get_state_dict as gsdt, set_state_dict as ssdt
from torch.distributed.checkpoint.stateful import Stateful as Sfsl
from torch.distributed.device_mesh import DeviceMesh as DvMs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP as FSP
from torch.distributed.fsdp import ShardingStrategy as ShrdStr
from torch.distributed.tensor import DTensor as DT
from torch.distributed.tensor.experimental import context_parallel as cp_ctx
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader as DLoader
from torch.utils.data.distributed import DistributedSampler as DSmp

from transformers import AutoModelForCausalLM as AML, AutoTokenizer as ATK


# Enable deterministic algorithms for reproducibility
torch.backends.cudnn.deterministic = True

# Configure logging system
_logging_module.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
log_handle = _logging_module.getLogger(__name__)


class ModelTrainer:
    """Main training orchestrator class."""

    @staticmethod
    def run():
        """Execute the full training workflow."""
        main()


def _initialize_environment():
    """Setup distributed computing environment parameters."""
    env_vars = {
        "TP_SIZE": "1",
        "DP_SIZE": "1", 
        "CP_SIZE": "1",
    }
    
    tP_Size = int(_os_module.environ.get("TP_SIZE", "1"))
    dP_Size = int(_os_module.environ.get("DP_SIZE", "1"))
    cP_Size = int(_os_module.environ.get("CP_SIZE", "1"))
    global_bs = 8
    seq_length = 1024
    num_train_steps = 10000
    learning_rate = 1e-5
    model_path = "HuggingFaceTB/SmolLM2-1.7B"
    
    chkpt_dir = f"chk_tp{tP_Size}_dp{dP_Size}_cp{cP_Size}"
    
    if "RANK" in _os_module.environ and "WORLD_SIZE" in _os_module.environ:
        distribs.init_process_group("nccl")
        rnk = distribs.get_rank()
        wrld_sz = distribs.get_world_size()
        local_rnk = int(_os_module.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rnk)
        
        assert wrld_sz == tP_Size * dP_Size * cP_Size, (
            f"World size ({wrld_sz}) != TP*DP*CP ({tP_Size}*{dP_Size}*{cP_Size})"
        )
        
        mesh = torch.arange(wrld_sz).reshape(dP_Size, tP_Size, cP_Size)
        wrld_msh = DvMs(device_type="cuda", mesh=mesh, mesh_dim_names=("dp", "tp", "cp"))
        tp_msh = wrld_msh["tp"]
        dp_msh = wrld_msh["dp"]
        cp_msh = wrld_msh["cp"]
        wrld_msh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
        
        log_handle.info(f"DeviceMesh created: {wrld_msh}")
        log_handle.info(
            f"Dist setup - Rnk: {rnk}, WS: {wrld_sz}, LR: {local_rnk}, DP: {dp_msh.get_local_rank()}, TP: {tp_msh.get_local_rank()}, CP: {cp_msh.get_local_rank()}"
        )
        
        if rnk == 0:
            wdb.init(
                project="tp_dp_exp",
                config={
                    "tP_Size": tP_Size,
                    "dP_Size": dP_Size,
                    "cP_Size": cP_Size,
                    "global_bs": global_bs,
                    "model_path": model_path,
                    "dataset_name": "roneneldan/TinyStories-1M",
                    "seq_length": seq_length,
                    "lr": learning_rate,
                    "weight_decay": 0.1,
                },
                name=f"tp{tP_Size}_dp{dP_Size}_cp{cP_Size}",
            )
            log_handle.info("Wandb initialized successfully.")


def _load_model_and_tokenizer(model_name):
    """Load pretrained model and tokenizer."""
    log_handle.info(f"Loading from {model_name}")
    tkr = ATK.from_pretrained(model_name)
    if tkr.pad_token is None:
        tkr.pad_token = tkr.eos_token
        log_handle.info(f"Set pad_token to eos_token: {tkr.pad_token}")

    model = AML.from_pretrained(
        model_name,
        device_mesh=tp_msh if distribs.is_initialized() else None,
        tp_plan="auto",
        dtype=torch.bfloat16,
    )
    log_handle.info(f"Model loaded: {tp_msh}")
    dev = torch.device(f"cuda:{local_rnk}")
    log_handle.info(f"Using device: {dev} for tensors")


def _load_dataset(dataset_name):
    """Load and process training dataset."""
    log_handle.info("Loading TinyStories...")
    raw_ds = dlsd(dataset_name, split="train[:1%]")
    
    def tokenz_fn(exmples):
        tok_batch = ATK(
            exmples["text"], padding=False, truncation=True, 
            max_length=seq_length, return_tensors=None
        )
        tok_batch["labels"] = tok_batch["input_ids"].copy()
        return tok_batch
    
    tok_ds = raw_ds.map(tokenz_fn, batched=True, remove_columns=["text"])
    log_handle.info(f"Dataset tokenized. Size: {len(tok_ds)}")
    
    def pack_seq_fn(exmples):
        all_toks = []
        for ids in exmples["input_ids"]:
            all_toks.extend(ids)
        
        num_seqs = len(all_toks) // (seq_length + 1)
        packed_in = []
        packed_lbls = []
        
        for i in range(num_seqs):
            st_idx = i * (seq_length + 1)
            end_idx = st_idx + (seq_length + 1)
            full_seq = all_toks[st_idx:end_idx]
            packed_in.append(full_seq[:-1])
            packed_lbls.append(full_seq[1:])
        
        return {"input_ids": packed_in, "labels": packed_lbls}
    
    pkd_ds = tok_ds.map(
        pack_seq_fn,
        batched=True,
        remove_columns=tok_ds.column_names,
        batch_size=1000,
        num_proc=60,
    )
    log_handle.info(f"Packed dataset. Size: {len(pkd_ds)}")
    
    pkd_ds = pkd_ds.shuffle(seed=42)
    log_handle.info("Shuffled packed dataset")
    
    return pkd_ds


def _create_dataloader(dataset):
    """Create data loader for training."""
    if distribs.is_initialized():
        assert global_bs % dp_msh.size() == 0, (
            f"Global batch ({global_bs}) not divisible by DP size ({dp_msh.size()})"
        )
        local_bs = global_bs // dp_msh.size()
    else:
        local_bs = global_bs
    
    log_handle.info(
        f"GBS: {global_bs}, DP: {dP_Size}, LBS: {local_bs}"
    )
    
    def collate_fn(batch):
        in_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        lbls = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"input_ids": in_ids, "labels": lbls}
    
    if distribs.is_initialized():
        smp = DSmp(
            dataset, num_replicas=dp_msh.size(), rank=dp_msh.get_local_rank(), shuffle=False
        )
    else:
        smp = None
    
    dl = DLoader(
        dataset,
        batch_size=local_bs,
        sampler=smp,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    log_handle.info(f"DataLoader ready. Distributed: {distribs.is_initialized()}")
    
    return dl


def _run_training(model, dataloader):
    """Execute the training loop."""
    optimizer = optims.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    log_handle.info(f"Training for {num_train_steps} steps...")
    model.train()
    step = 0
    
    while step < num_train_steps:
        for batch in dataloader:
            if step >= num_train_steps:
                break
            
            batch = {k: v.to(dev) for k, v in batch.items()}
            optimizer.zero_grad()
            
            bs = batch["input_ids"].shape[0]
            pos_ids = torch.arange(0, seq_length, dtype=torch.long, device=dev)
            pos_ids = pos_ids.unsqueeze(0).expand(bs, -1)
            batch["position_ids"] = pos_ids
            
            from torch.distributed.tensor.experimental._attention import _cp_options
            _cp_options.enable_load_balance = False
            
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                cp_cntx = (
                    NulCtxxt()
                    if cp_msh.size() == 1
                    else cp_ctx(
                        cp_msh,
                        buffers=[
                            batch["input_ids"],
                            batch["labels"],
                            batch["position_ids"],
                        ],
                        buffer_seq_dims=[1, 1, 1],
                    )
                )
                
                with cp_cntx:
                    lbls = batch.pop("labels")
                    outputs = model(**batch)
                    loss = outputs.loss
                    logts = outputs.logits
                    
                    loss = model.loss_function(
                        logts=logts, lbls=None, shift_labels=lbls, 
                        vocab_sz=model.config.vocab_size
                    )
                    loss.backward()
                    
                all_reduce_grads(model, wrld_msh)
                
                if hasattr(model, "clip_grad_norm_"):
                    gradnrml = model.clip_grad_norm_(max_norm=1.0, norm_type=2.0)
                else:
                    assert len(list(model.parameters())) > 5, "No params found"
                    gradnrml = clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0, foreach=True)
                
                optimizer.step()
                
                if distribs.is_initialized() and (cp_msh.size() > 1 or dp_msh.size() > 1):
                    distribs.all_reduce(loss, group=wrld_msh["dp_cp"].get_group(), op=distribs.ReduceOp.AVG)
                curr_loss = loss.item()
                
                if not distribs.is_initialized() or distribs.get_rank() == 0:
                    log_handle.info(
                        f"Step {step} | GBS {global_bs} | DP {dp_msh.size()} | TP {tp_msh.size()} | CP {cp_msh.size()} | Loss {curr_loss:.4f} | Gradnorm {gradnrml:.4f}"
                    )
                    wdb.log(
                        {
                            "train/loss": curr_loss,
                            "train/gradnorm": gradnrml,
                            "step": step,
                            "lr": learning_rate,
                            "GBS": global_bs,
                        }
                    )
            
            step += 1
    
    log_handle.info("Training complete.")


def all_reduce_grads(model, world_mesh, use_ddp):
    """Reduce gradients across distributed groups."""
    cp_msh = world_mesh["cp"]
    if use_ddp:
        mesh = cp_msh
    else:
        mesh = world_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
    
    if distribs.is_initialized() and mesh.size() > 1:
        for name, param in model.named_parameters():
            if param.grad is not None:
                if isinstance(param.grad, DT):
                    loc_grad = param.grad.to_local()
                    torch.distributed.all_reduce(loc_grad, op=torch.distributed.ReduceOp.SUM, group=mesh.get_group())
                    loc_grad = loc_grad / mesh.size()
                    param.grad = DT.from_local(
                        loc_grad, device_mesh=param.grad.device_mesh, placements=param.grad.placements
                    )
                else:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG, group=mesh.get_group())


class AppState(Sfsl):
    """Stateful checkpoint wrapper."""

    def __init__(self, model, opt=None):
        self.model = model
        self.opt = opt

    def state_dict(self):
        mdl_sd, opt_sd = gsdt(self.model, self.opt)
        return {"model": mdl_sd, "optim": opt_sd}

    def load_state_dict(self, state_dict):
        ssdt(
            self.model, self.opt, model_state_dict=state_dict["model"], 
            optim_state_dict=state_dict["optim"]
        )


def clip_grad_norm_(params, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """Clip gradient norms for stability."""
    params = [p for p in params if p.grad is not None]
    assert len(params) > 0, "No parameters with gradients"

    if norm_type == float("inf"):
        tot_norm = max(p.grad.detach().abs().max() for p in params)
    else:
        tot_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]), norm_type)

    if isinstance(tot_norm, DT):
        tot_norm = tot_norm.full_tensor()

    clip_coef = max_norm / (tot_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad.detach().mul_(clip_coef)

    return tot_norm


def main():
    """Main entry point."""
    _initialize_environment()
    
    _load_model_and_tokenizer(model_path)
    
    dataset = _load_dataset("roneneldan/TinyStories-1M")
    dataloader = _create_dataloader(dataset)
    
    _run_training(model, dataloader)
    
    if distribs.is_initialized():
        state_dict = {"app": AppState(model)}
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=chkpt_dir,
        )
        log_handle.info(f"Checkpoint saved to {chkpt_dir}")
    else:
        save_path = "test_model_nondist"
        model.save_pretrained(save_path, safe_serialization=False)
        ATK.save_pretrained(save_path)
        log_handle.info(f"Saved to {save_path}")
    
    distribs.destroy_process_group()
    log_handle.info("Distributed process group cleaned up")
    
    if distribs.get_rank() == 0:
        wdb.finish()
        log_handle.info("Wandb run finished.")


if __name__ == "__main__":
    ModelTrainer.run()