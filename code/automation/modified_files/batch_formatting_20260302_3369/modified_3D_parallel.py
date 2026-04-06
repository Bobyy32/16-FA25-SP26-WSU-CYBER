# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module serves as a training validation harness employing Tensor Parallelism along with Data Parallelism techniques.

Usage:
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=5,6,7
TP_SIZE=2 DP_SIZE=2 torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29503 examples/3D_parallel.py
CP_SIZE=2 DP_SIZE=2 torchrun --nproc_per_node=4 examples/3D_parallel.py
CP_SIZE=2 TP_SIZE=2 torchrun --nproc_per_node=4 examples/3D_parallel.py
DP_SIZE=2 CP_SIZE=2 TP_SIZE=2 torchrun --nproc_per_node=8 examples/3D_parallel.py

TP_SIZE=1 CP_SIZE=4 torchrun --nproc_per_node=4 examples/3D_parallel.py
TP_SIZE=1 DP_SIZE=4 torchrun --nproc_per_node=4 examples/3D_parallel.py
TP_SIZE=4 DP_SIZE=1 torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29503 examples/3D_parallel.py
IGNORE_SANITY=1 CP_SIZE=1 TP_SIZE=1 DP_SIZE=1 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:29504 examples/3D_parallel.py
localhost:29504 test_train.py
"""

import logging as log
from collections.abc import Iterable as IterableType
from contextlib import nullcontext as null_ctx

import torch as pytorch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.optim as optim
import wandb as wandb_module
from datasets import load_dataset as get_ds
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import context_parallel as cp_ctx
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader as DataLoad
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM as LMModel, AutoTokenizer as Tokenizer


log.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=log.INFO,
)

logger = log.getLogger(__name__)

# Set deterministic backends appropriately
pytorch.backends.cudnn.deterministic = True


def main():
    tp_size = int(log.getenv("TP_SIZE", "1"))
    dp_size = int(log.getenv("DP_SIZE", "1"))
    cp_size = int(log.getenv("CP_SIZE", "1"))

    sdpa_backend = SDPBackend.FLASH_ATTENTION
    global_batch_size = 8
    seq_len = 1024
    num_train_steps = 10000
    LR = 1e-5
    model_ref = "HuggingFaceTB/SmolLM2-1.7B"

    CHECKPOINT_DIR = f"checkpoint_tp{tp_size}_dp{dp_size}_cp{cp_size}"

    # Setup distributed communication environment
    if "RANK" in log.getenv() and "WORLD_SIZE" in log.getenv():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(log.getenv("LOCAL_RANK"))
        pytorch.cuda.set_device(local_rank)

        assert world_size == tp_size * dp_size * cp_size, (
            f"World size ({world_size}) must equal TP size ({tp_size}) * DP size ({dp_size}) * CP size ({cp_size})"
        )

        mesh = pytorch.arange(world_size).reshape(dp_size, tp_size, cp_size)
        world_mesh = DeviceMesh(device_type="cuda", mesh=mesh, mesh_dim_names=("dp", "tp", "cp"))
        tp_mesh = world_mesh["tp"]
        dp_mesh = world_mesh["dp"]
        cp_mesh = world_mesh["cp"]
        world_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
        logger.info(f"Created DeviceMesh: {world_mesh}")
        logger.info(
            f"Distributed setup - Rank: {rank}, World size: {world_size}, Local rank: {local_rank}, DP: {dp_mesh.get_local_rank()}, TP: {tp_mesh.get_local_rank()}, CP: {cp_mesh.get_local_rank()}"
        )

        if dist.get_rank() == 0:
            wandb_module.init(
                project="tp_dp_test",
                config={
                    "tp_size": tp_size,
                    "dp_size": dp_size,
                    "cp_size": cp_size,
                    "global_batch_size": global_batch_size,
                    "model_name": model_ref,
                    "dataset": "roneneldan/TinyStories-1M",
                    "seq_len": seq_len,
                    "lr": LR,
                    "weight_decay": 0.1,
                },
                name=f"llama_tp{tp_size}_dp{dp_size}_cp{cp_size}"
                if model_ref == "unsloth/Llama-3.2-1B"
                else f"tp{tp_size}_dp{dp_size}_cp{cp_size}",
            )
            logger.info("Wandb initialized.")

    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer from {model_ref}")
    tokenizer = Tokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    model_ref = LMModel.from_pretrained(
        model_ref,
        device_mesh=tp_mesh if dist.is_initialized() else None,
        tp_plan="auto",
        dtype=pytorch.bfloat16,
    )
    logger.info(f"Model loaded onto device mesh: {tp_mesh}")
    device = pytorch.device(f"cuda:{local_rank}")
    logger.info(f"Using device: {device} for non-model tensors")
    use_ddp = False
    if dist.is_initialized() and dp_mesh.size() > 1:
        model_ref = FSDP(model_ref, device_mesh=dp_mesh, sharding_strategy=ShardingStrategy.NO_SHARD)
        use_ddp = True

    model_ref.train()

    logger.info("Loading TinyStories dataset...")
    raw_dataset = get_ds("roneneldan/TinyStories", split="train[:1%]")

    def tokenize_function(examples):
        tokenized_batch = Tokenizer(
            examples["text"], padding=False, truncation=True, max_length=seq_len, return_tensors=None
        )
        tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()
        return tokenized_batch

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    logger.info(f"Dataset loaded and tokenized. Size: {len(tokenized_dataset)}")

    def create_packed_sequences(examples):
        all_tokens = []
        for input_ids in examples["input_ids"]:
            all_tokens.extend(input_ids)

        num_sequences = len(all_tokens) // (seq_len + 1)
        packed_input_ids = []
        packed_labels = []

        for i in range(num_sequences):
            start_idx = i * (seq_len + 1)
            end_idx = start_idx + (seq_len + 1)
            full_sequence = all_tokens[start_idx:end_idx]
            packed_input_ids.append(full_sequence[:-1])
            packed_labels.append(full_sequence[1:])

        return {"input_ids": packed_input_ids, "labels": packed_labels}

    packed_dataset = tokenized_dataset.map(
        create_packed_sequences,
        batched=True,
        remove_columns=tokenized_dataset.column_names,
        batch_size=1000,
        num_proc=60,
    )
    logger.info(f"Dataset packed. New size: {len(packed_dataset)}")

    packed_dataset = packed_dataset.shuffle(seed=42)
    logger.info("Packed dataset shuffled")

    if dist.is_initialized():
        assert global_batch_size % dp_mesh.size() == 0, (
            f"Global batch size ({global_batch_size}) must be divisible by DP size ({dp_mesh.size()})"
        )
        local_batch_size = global_batch_size // dp_mesh.size()
    else:
        local_batch_size = global_batch_size

    logger.info(
        f"Global batch size: {global_batch_size}, DP size: {dp_size if dist.is_initialized() else 1}, Local batch size: {local_batch_size}"
    )

    def collate_fn(batch):
        input_ids = pytorch.tensor([item["input_ids"] for item in batch], dtype=pytorch.long)
        labels = pytorch.tensor([item["labels"] for item in batch], dtype=pytorch.long)
        return {"input_ids": input_ids, "labels": labels}

    if dist.is_initialized():
        sampler = DistributedSampler(
            packed_dataset, num_replicas=dp_mesh.size(), rank=dp_mesh.get_local_rank(), shuffle=False
        )
    else:
        sampler = None

    dataloader = DataLoad(
        packed_dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    logger.info(f"DataLoader created. Distributed: {dist.is_initialized()}")

    optimizer = optim.AdamW(model_ref.parameters(), lr=LR, weight_decay=0.1)

    # Training loop
    logger.info(f"Starting training for {num_train_steps} steps...")
    model_ref.train()
    step = 0
    while step < num_train_steps:
        for batch in dataloader:
            if step >= num_train_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            batch_size = batch["input_ids"].shape[0]
            position_ids = pytorch.arange(0, seq_len, dtype=pytorch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            batch["position_ids"] = position_ids

            _cp_options.enable_load_balance = False

            with sdpa_kernel(sdpa_backend):
                cp_context = (
                    null_ctx()
                    if cp_mesh.size() == 1
                    else cp_ctx(
                        cp_mesh,
                        buffers=[
                            batch["input_ids"],
                            batch["labels"],
                            batch["position_ids"],
                        ],
                        buffer_seq_dims=[1, 1, 1],
                    )
                )
                with cp_context:
                    labels = batch.pop("labels")
                    outputs = model_ref(**batch)
                    loss = outputs.loss
                    logits = outputs.logits

                    loss = model_ref.loss_function(
                        logits=logits, labels=None, shift_labels=labels, vocab_size=model_ref.config.vocab_size
                    )
                    loss.backward()

                all_reduce_grads(model_ref, world_mesh, use_ddp=use_ddp)

                if hasattr(model_ref, "clip_grad_norm_"):
                    gradnorm = model_ref.clip_grad_norm_(max_norm=1.0, norm_type=2.0)
                else:
                    assert len(list(model_ref.parameters())) > 5, "No parameters found in model"
                    gradnorm = clip_grad_norm_(model_ref.parameters(), max_norm=1.0, norm_type=2.0, foreach=True)

                optimizer.step()
                
                if dist.is_initialized() and (cp_mesh.size() > 1 or dp_mesh.size() > 1):
                    dist.all_reduce(loss, group=world_mesh["dp_cp"].get_group(), op=dist.ReduceOp.AVG)
                current_loss = loss.item()

                if not dist.is_initialized() or dist.get_rank() == 0:
                    logger.info(
                        f"Step: {step} | GBS: {global_batch_size} | DP: {dp_mesh.size()} | TP: {tp_mesh.size()} | CP: {cp_mesh.size()} | Loss: {current_loss} | Gradnorm: {gradnorm} | lr: {LR}"
                    )
                    wandb_module.log(
                        {
                            "train/loss": current_loss,
                            "train/gradnorm": gradnorm,
                            "step": step,
                            "lr": LR,
                            "GBS": global_batch_size,
                        }
                    )

            step += 1

    logger.info("Training loop finished.")

    # Save model using DCP
    if dist.is_initialized():
        state_dict = {"app": AppState(model_ref, optimizer)}
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=CHECKPOINT_DIR,
        )
        logger.info(f"Saved checkpoint to {CHECKPOINT_DIR}")
    else:
        save_dir = "test_model_nondist"
        model_ref.save_pretrained(save_dir, safe_serialization=False)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved model to {save_dir}")

    dist.destroy_process_group()
    logger.info("Cleaned up distributed process group")
    
    if dist.get_rank() == 0:
        wandb_module.finish()
        logger.info("Wandb run finished.")


def all_reduce_grads(model_ref, world_mesh, use_ddp):
    """All reduce gradients across dp_cp if applicable."""
    cp_mesh = world_mesh["cp"]
    if use_ddp:
        mesh = cp_mesh
    else:
        mesh = world_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
    if dist.is_initialized() and mesh.size() > 1:
        for name, param in model_ref.named_parameters():
            if param.grad is not None:
                if isinstance(param.grad, DTensor):
                    local_grad = param.grad.to_local()
                    torch.distributed.all_reduce(local_grad, op=torch.distributed.ReduceOp.SUM, group=mesh.get_group())
                    local_grad = local_grad / mesh.size()
                    param.grad = DTensor.from_local(
                        local_grad, device_mesh=param.grad.device_mesh, placements=param.grad.placements
                    )
                else:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG, group=mesh.get_group())


class AppState(Stateful):
    """Wrapper for checkpointing the Application State including model and optimizer."""

    def __init__(self, model_ref, optimizer=None):
        self.model = model_ref
        self.optimizer = optimizer

    def state_dict(self):
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, self.optimizer, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optim"]
        )


def clip_grad_norm_(
    parameters: IterableType[pytorch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> pytorch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.
    """
    parameters = [p for p in parameters if p.grad is not None]
    assert len(parameters) > 0, "No parameters with gradients found"

    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = pytorch.norm(pytorch.stack([pytorch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)

    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    return total_norm


if __name__ == "__main__":
    main()