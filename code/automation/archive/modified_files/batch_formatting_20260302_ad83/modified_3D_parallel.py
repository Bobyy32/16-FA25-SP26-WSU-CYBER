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

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.optim as optim
import wandb
from collections.abc import Iterable
from contextlib import nullcontext
from datasets import load_dataset
from logging import basicConfig, getLogger
from os import environ, getcwd, listdir, path, system
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

__doc__ = """
This script is used to test training a model using Tensor Parallelism and Data Parallelism.

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

# Configure logging system
basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=getLogger(__name__).INFO,
)
_logging_handler = getLogger(__name__)

# Disable cuDNN non-determinism for reproducible training
torch.backends.cudnn.deterministic = True


def synchronize_gradient_across_shards(model_instance, mesh_grid, distributed_dp):
    """Synchronize gradients across dp_cp if applicable."""
    context_parallel_mesh = mesh_grid["cp"]
    if distributed_dp:
        # DDP/FSDP manages gradient synchronization automatically
        mesh_used = context_parallel_mesh
    else:
        flattened_mesh = mesh_grid["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
        mesh_used = flattened_mesh
    if dist.is_initialized() and mesh_used.size() > 1:
        for param_name, param_tensor in model_instance.named_parameters():
            if param_tensor.grad is not None:
                # Handle DTensor gradients specially
                if isinstance(param_tensor.grad, DTensor):
                    local_gradient = param_tensor.grad.to_local()
                    torch.distributed.all_reduce(
                        local_gradient,
                        op=torch.distributed.ReduceOp.SUM,
                        group=mesh_used.get_group()
                    )
                    local_gradient = local_gradient / mesh_used.size()
                    param_tensor.grad = DTensor.from_local(
                        local_gradient,
                        device_mesh=param_tensor.grad.device_mesh,
                        placements=param_tensor.grad.placements
                    )
                else:
                    # Regular tensor gradient reduction
                    torch.distributed.all_reduce(
                        param_tensor.grad,
                        op=torch.distributed.ReduceOp.AVG,
                        group=mesh_used.get_group()
                    )


class CheckpointContainer(Stateful):
    """Holds Application State for checkpointing including neural net and optimizer."""

    def __init__(self, nn_model, optim=None):
        self.nn_model = nn_model
        self.optim = optim

    def export_state_dict(self):
        nn_sd, opt_sd = get_state_dict(self.nn_model, self.optim)
        return {"model": nn_sd, "optim": opt_sd}

    def import_state_dict(self, state_dict_data):
        set_state_dict(
            self.nn_model,
            self.optim,
            model_state_dict=state_dict_data["model"],
            optim_state_dict=state_dict_data["optim"]
        )


def assemble_concatenated_batches(examples_batch):
    """Create packed sequences from examples."""
    # Flatten all token sequences
    flattened_tokens = []
    for input_ids in examples_batch["input_ids"]:
        flattened_tokens.extend(input_ids)

    # Split into sequence chunks
    chunk_count = len(flattened_tokens) // (1025)  # seq_len + 1
    packed_inps = []
    packed_labels = []

    for i in range(chunk_count):
        start_idx = i * 1025
        end_idx = start_idx + 1025
        full_seq = flattened_tokens[start_idx:end_idx]
        # Remove last token for inputs
        packed_inps.append(full_seq[:-1])
        # Remove first token for labels
        packed_labels.append(full_seq[1:])

    return {"input_ids": packed_inps, "labels": packed_labels}


def prepare_batch_tensor_structure(batch_list):
    """Convert batch list to tensors."""
    inputs = torch.tensor([item["input_ids"] for item in batch_list], dtype=torch.long)
    targets = torch.tensor([item["labels"] for item in batch_list], dtype=torch.long)
    return {"input_ids": inputs, "labels": targets}


def create_tokenized_sequences(examples_batch):
    """Tokenize text without padding."""
    tokenized_out = __import__('transformers').AutoTokenizer.from_pretrained(__import__('sys').argv[1])
    batch = tokenizer(
        examples_batch["text"],
        padding=False,
        truncation=True,
        max_length=1024,
        return_tensors=None
    )
    batch["labels"] = batch["input_ids"].copy()
    return batch


def _cp_options_setter(value):
    """Configure attention load balance options."""
    if hasattr(__import__('torch.distributed.tensor.experimental._attention'), '_cp_options'):
        __import__('torch.distributed.tensor.experimental._attention')._cp_options.enable_load_balance = value


# Main training entry point
def orchestration_entry_point():
    parallel_tensor_size = int(environ.get("TP_SIZE", "1"))
    parallel_device_size = int(environ.get("DP_SIZE", "1"))
    parallel_context_size = int(environ.get("CP_SIZE", "1"))  # Add CP size configuration
    sdpa_backend_type = SDPBackend.FLASH_ATTENTION  # For CP
    # sdpa_backend_type = SDPBackend.MATH # For CP
    total_batch_dimension = 8  # Desired global batch size
    sequence_length_param = 1024  # Sequence length
    training_iteration_count = 10000  # Number of training steps
    learning_rate_value = 1e-5
    pretrained_model_identifier = "HuggingFaceTB/SmolLM2-1.7B"
    # pretrained_model_identifier = "unsloth/Llama-3.2-1B"

    save_directory_path = f"checkpoint_tp{parallel_tensor_size}_dp{parallel_device_size}_cp{parallel_context_size}"

    # Initialize distributed environment
    if "RANK" in environ and "WORLD_SIZE" in environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        assert world_size == parallel_tensor_size * parallel_device_size * parallel_context_size, (
            f"World size ({world_size}) must equal TP size ({parallel_tensor_size}) * DP size ({parallel_device_size}) * CP size ({parallel_context_size})"
        )

        mesh = torch.arange(world_size).reshape(parallel_device_size, parallel_tensor_size, parallel_context_size)
        world_mesh = DeviceMesh(device_type="cuda", mesh=mesh, mesh_dim_names=("dp", "tp", "cp"))
        tp_mesh = world_mesh["tp"]
        dp_mesh = world_mesh["dp"]
        cp_mesh = world_mesh["cp"]
        world_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
        _logging_handler.info(f"Created DeviceMesh: {world_mesh}")
        _logging_handler.info(
            f"Distributed setup - Rank: {rank}, World size: {world_size}, Local rank: {local_rank}, DP: {dp_mesh.get_local_rank()}, TP: {tp_mesh.get_local_rank()}, CP: {cp_mesh.get_local_rank()}"
        )

        if dist.get_rank() == 0:
            wandb.init(
                project="tp_dp_test",
                config={
                    "parallel_tensor_size": parallel_tensor_size,
                    "parallel_device_size": parallel_device_size,
                    "parallel_context_size": parallel_context_size,
                    "total_batch_dimension": total_batch_dimension,
                    "pretrained_model_identifier": pretrained_model_identifier,
                    "dataset": "roneneldan/TinyStories-1M",
                    "sequence_length_param": sequence_length_param,
                    "learning_rate_value": learning_rate_value,
                    "weight_decay": 0.1,
                },
                name=f"llama_tp{parallel_tensor_size}_dp{parallel_device_size}_cp{parallel_context_size}"
                if pretrained_model_identifier == "unsloth/Llama-3.2-1B"
                else f"tp{parallel_tensor_size}_dp{parallel_device_size}_cp{parallel_context_size}",
            )
            _logging_handler.info("Wandb initialized.")
            # Log the current file to wandb
            wandb.save("test_train.py")

    # Load model and tokenizer
    _logging_handler.info(f"Loading model and tokenizer from {pretrained_model_identifier}")
    tokenizer = __import__('transformers').AutoTokenizer.from_pretrained(pretrained_model_identifier)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        _logging_handler.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    nn_model = __import__('transformers').AutoModelForCausalLM.from_pretrained(
        pretrained_model_identifier,
        device_mesh=tp_mesh if dist.is_initialized() else None,
        tp_plan="auto",
        dtype=torch.bfloat16,
    )
    _logging_handler.info(f"Model loaded onto device mesh: {tp_mesh}")
    device = torch.device(f"cuda:{local_rank}")
    _logging_handler.info(f"Using device: {device} for non-model tensors")
    use_ddp = False
    if dist.is_initialized() and dp_mesh.size() > 1:
        nn_model = FSDP(nn_model, device_mesh=dp_mesh, sharding_strategy=ShardingStrategy.NO_SHARD)
        use_ddp = True

    nn_model.train()

    _logging_handler.info("Loading TinyStories dataset...")
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")  # Use 1% for faster testing

    tokenized_dataset = raw_dataset.map(create_tokenizer_sequences, batched=True, remove_columns=["text"])
    _logging_handler.info(f"Dataset loaded and tokenized. Size: {len(tokenized_dataset)}")

    # Apply packing to the dataset
    packed_dataset = tokenized_dataset.map(
        assemble_concatenated_batches,
        batched=True,
        remove_columns=tokenized_dataset.column_names,
        batch_size=1000,  # Process in batches for efficiency
        num_proc=60,
    )
    _logging_handler.info(f"Dataset packed. New size: {len(packed_dataset)}")

    # Shuffle the packed dataset
    packed_dataset = packed_dataset.shuffle(seed=42)
    _logging_handler.info("Packed dataset shuffled")

    # Calculate local batch dimension
    if dist.is_initialized():
        assert total_batch_dimension % dp_mesh.size() == 0, (
            f"Global batch size ({total_batch_dimension}) must be divisible by DP size ({dp_mesh.size()})"
        )
        per_process_batch_dimension = total_batch_dimension // dp_mesh.size()
    else:
        per_process_batch_dimension = total_batch_dimension

    _logging_handler.info(
        f"Total batch dimension: {total_batch_dimension}, DP size: {parallel_device_size if dist.is_initialized() else 1}, Local batch dimension: {per_process_batch_dimension}"
    )

    # Simple collate function since sequences are already packed
    if dist.is_initialized():
        sampler = DistributedSampler(
            packed_dataset, num_replicas=dp_mesh.size(), rank=dp_mesh.get_local_rank(), shuffle=False
        )
    else:
        sampler = None

    dataloader = DataLoader(
        packed_dataset,
        batch_size=per_process_batch_dimension,
        sampler=sampler,
        shuffle=False,
        collate_fn=prepare_batch_tensor_structure,
        pin_memory=True,
    )
    _logging_handler.info(f"DataLoader created. Distributed: {dist.is_initialized()}")

    parameter_update_engine = optim.AdamW(nn_model.parameters(), lr=learning_rate_value, weight_decay=0.1)

    # Training loop
    _logging_handler.info(f"Starting training for {training_iteration_count} steps...")
    nn_model.train()
    step = 0
    while step < training_iteration_count:
        for batch in dataloader:
            if step >= training_iteration_count:
                break  # Exit loop if max steps reached

            # Move batch to appropriate device
            batch = {k: v.to(device) for k, v in batch.items()}
            parameter_update_engine.zero_grad()

            # Add position_ids to batch before CP sharding
            batch_size = batch["input_ids"].shape[0]
            position_ids = torch.arange(0, sequence_length_param, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            batch["position_ids"] = position_ids

            with sdpa_kernel(sdpa_backend_type):  # TODO: ideally move this to attention implementation
                cp_context = (
                    nullcontext()
                    if cp_mesh.size() == 1
                    else context_parallel(
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
                    # Pop labels from batch before model forward pass
                    labels = batch.pop("labels")
                    outputs = nn_model(**batch)  # [mbs, seq_len/cp]
                    loss = outputs.loss
                    logits = outputs.logits

                    # Compute loss with shifted labels
                    loss = nn_model.loss_function(
                        logits=logits, labels=None, shift_labels=labels, vocab_size=nn_model.config.vocab_size
                    )
                    loss.backward()

                # all reduce grads across dp_cp if applicable
                synchronize_gradient_across_shards(nn_model, world_mesh, use_ddp)

                if hasattr(nn_model, "clip_grad_norm_"):
                    gradnorm = nn_model.clip_grad_norm_(max_norm=1.0, norm_type=2.0)  # TODO: fix reported gradnorm
                else:
                    # only works with FSDP's NO_SHARD otherwise we should use FSDP's clip_grad_norm_
                    assert len(list(nn_model.parameters())) > 5, "No parameters found in model. Probably DDP bug.."
                    gradnorm = clip_gradient_norm_(nn_model.parameters(), max_norm=1.0, norm_type=2.0, foreach=True)

                parameter_update_engine.step()
                # allreduce loss across cp_dp before logging
                if dist.is_initialized() and (cp_mesh.size() > 1 or dp_mesh.size() > 1):
                    dist.all_reduce(loss, group=world_mesh["dp_cp"].get_group(), op=dist.ReduceOp.AVG)
                current_loss = loss.item()

                # Log loss and gradnorm to wandb (only on rank 0 of dp group)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    _logging_handler.info(
                        f"Step: {step} | GBS: {total_batch_dimension} | DP: {dp_mesh.size()} | TP: {tp_mesh.size()} | CP: {cp_mesh.size()} | Loss: {current_loss} | Gradnorm: {gradnorm} | lr: {learning_rate_value}"
                    )
                    wandb.log(
                        {
                            "train/loss": current_loss,
                            "train/gradnorm": gradnorm,
                            "step": step,
                            "lr": learning_rate_value,
                            "GBS": total_batch_dimension,
                        }
                    )

            step += 1  # Increment step count

    _logging_handler.info("Training loop finished.")

    # Save model using DCP (only if distributed)
    if dist.is_initialized():
        state_dict = {"app": CheckpointContainer(nn_model, parameter_update_engine)}
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=save_directory_path,
        )
        _logging_handler.info(f"Saved checkpoint to {save_directory_path}")
    else:
        # Fallback to regular save for non-distributed case
        save_dir = "test_model_nondist"
        nn_model.save_pretrained(save_dir, safe_serialization=False)
        tokenizer.save_pretrained(save_dir)  # Save tokenizer too
        _logging_handler.info(f"Saved model to {save_dir}")

    dist.destroy_process_group()
    _logging_handler.info("Cleaned up distributed process group")
    # Finish wandb run on rank 0
    if dist.get_rank() == 0:
        wandb.finish()
        _logging_handler.info("Wandb run finished.")


def clip_gradient_norm_(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.
    
    Args:
        parameters: Iterable of tensors with gradients to clip
        max_norm: Maximum allowed norm for gradients
        norm_type: Norm type (L2 default)
        error_if_nonfinite: Raise error if norm is non-finite
        foreach: Use foreach implementation if True
    
    Returns:
        Total gradient norm
    """
    # Filter out parameters with no gradients
    parameters = [p for p in parameters if p.grad is not None]
    assert len(parameters) > 0, "No parameters with gradients found"

    # Calculate total norm
    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)

    # Convert DTensor to local tensor if needed
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    return total_norm


if __name__ == "__main__":
    orchestration_entry_point()