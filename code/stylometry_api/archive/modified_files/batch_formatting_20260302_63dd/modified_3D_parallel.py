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

r"""
This script enables multi-device training with Tensor Parallelism (TP), Data Parallelism (DP),
and Context Parallelism (CP) configurations. 

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

from collections.abc import Iterable as IterableABC
import logging
import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.optim as optim
import wandb
from datasets import load_dataset
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful as StatefulABC
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy as ShardingStrategyABC
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import context_parallel
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Relative import attempt for internal module structure (maintained functionally)
try:
    from .torch.distributed.tensor.experimental._attention import _cp_options
except ImportError:
    pass


def __main__():
    """
    Entry point function to orchestrate distributed model training.
    Initializes TP/DP/CP mesh configurations, loads datasets and models,
    executes the training loop with checkpointing support.
    
    Args: None - configuration derived from environment variables
    
    Returns: None after completing training or raising exception
    """
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    dp_size = int(os.environ.get("DP_SIZE", "1"))
    cp_size = int(os.environ.get("CP_SIZE", "1"))  # CP size configuration parameter
    sdpa_backend = SDPBackend.FLASH_ATTENTION  # For CP operations
    global_batch_size = 8  # Desired global batch dimension
    seq_len = 1024  # Sequence length for processing
    num_train_steps = 10000  # Training iterations count
    LR = 1e-5
    model_name = "HuggingFaceTB/SmolLM2-1.7B"

    CHECKPOINT_DIR = f"checkpoint_tp{tp_size}_dp{dp_size}_cp{cp_size}"

    # Initialize distributed process group
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        assert world_size == tp_size * dp_size * cp_size, (
            f"World size ({world_size}) must equal TP size ({tp_size}) * DP size ({dp_size}) * CP size ({cp_size})"
        )

        mesh = torch.arange(world_size).reshape(dp_size, tp_size, cp_size)
        world_mesh = DeviceMesh(device_type="cuda", mesh=mesh, mesh_dim_names=("dp", "tp", "cp"))
        tp_mesh = world_mesh["tp"]
        dp_mesh = world_mesh["dp"]
        cp_mesh = world_mesh["cp"]
        world_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")

    # Initialize logging system
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Initialize Weights and Biases tracking if distributed
    if dist.get_rank() == 0:
        wandb.init(
            project="tp_dp_test",
            config={
                "tp_size": tp_size,
                "dp_size": dp_size,
                "cp_size": cp_size,
                "global_batch_size": global_batch_size,
                "model_name": model_name,
                "dataset": "roneneldan/TinyStories-1M",
                "seq_len": seq_len,
                "lr": LR,
                "weight_decay": 0.1,
            },
            name=f"llama_tp{tp_size}_dp{dp_size}_cp{cp_size}"
            if model_name == "unsloth/Llama-3.2-1B"
            else f"tp{tp_size}_dp{dp_size}_cp{cp_size}",
        )

    # Load preprocessing components
    logger.info(f"Loading model and tokenizer from {model_name}")
    tokenizer = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Initialize model with device mesh support
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_mesh=tp_mesh if dist.is_initialized() else None,
        tp_plan="auto",
        dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded onto device mesh: {tp_mesh}")
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Using device: {device} for non-model tensors")

    # Setup distributed parallelism strategy
    use_ddp = False
    if dist.is_initialized() and dp_mesh.size() > 1:
        model = FSDP(model, device_mesh=dp_mesh, sharding_strategy=ShardingStrategy.NO_SHARD)
        use_ddp = True

    model.train()

    # Load and prepare training dataset
    logger.info("Loading TinyStories dataset...")
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")  # Use 1% for faster testing

    def tokenize_function(examples):
        """
        Tokenize input text samples without explicit padding.
        
        Args: examples (dict) containing 'text' key
        
        Returns: dict with tokenized features and labels
        """
        tokenized_batch = tokenizer(
            examples["text"], padding=False, truncation=True, max_length=seq_len, return_tensors=None
        )
        tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()
        return tokenized_batch

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    logger.info(f"Dataset loaded and tokenized. Size: {len(tokenized_dataset)}")

    # Create packed sequences for efficient batch processing
    def create_packed_sequences(examples):
        """
        Concatenate all tokens and split into fixed-length sequence chunks.
        
        Args: examples (dict) with 'input_ids' containing list of token arrays
        
        Returns: dict with packed 'input_ids' and shifted 'labels'
        """
        # Flatten all sequences
        all_tokens = []
        for input_ids in examples["input_ids"]:
            all_tokens.extend(input_ids)

        # Split into sequences of seq_len + 1 (for input + label)
        num_sequences = len(all_tokens) // (seq_len + 1)
        packed_input_ids = [
            full_sequence[:-1]
            for i, full_sequence in enumerate(
                all_tokens[i * (seq_len + 1):(i + 1) * (seq_len + 1)]
                for i in range(num_sequences)
            )
        ]
        packed_labels = [
            full_sequence[1:]
            for i, full_sequence in enumerate(
                all_tokens[i * (seq_len + 1):(i + 1) * (seq_len + 1)]
                for i in range(num_sequences)
            )
        ]

        return {"input_ids": packed_input_ids, "labels": packed_labels}

    # Apply packing transformation to dataset
    packed_dataset = tokenized_dataset.map(
        create_packed_sequences,
        batched=True,
        remove_columns=tokenized_dataset.column_names,
        batch_size=1000,  # Process in batches for efficiency
        num_proc=60,
    )
    logger.info(f"Dataset packed. New size: {len(packed_dataset)}")

    # Shuffle the packed dataset
    packed_dataset = packed_dataset.shuffle(seed=42)
    logger.info("Packed dataset shuffled")

    # Calculate local batch dimensions per device
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

    # Collate function for packed sequences (already formatted)
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    # Setup distributed sampler if needed
    if dist.is_initialized():
        sampler = DistributedSampler(
            packed_dataset, num_replicas=dp_mesh.size(), rank=dp_mesh.get_local_rank(), shuffle=False
        )
    else:
        sampler = None

    dataloader = DataLoader(
        packed_dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    logger.info(f"DataLoader created. Distributed: {dist.is_initialized()}")

    # Initialize optimizer for model parameters
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    # Execute training iterations
    logger.info(f"Starting training for {num_train_steps} steps...")
    model.train()
    step = 0
    while step < num_train_steps:
        for batch in dataloader:
            if step >= num_train_steps:
                break  # Exit loop if max steps reached

            # Transfer batch to appropriate device
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            # Add position_ids to batch before CP sharding
            batch_size = batch["input_ids"].shape[0]
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            batch["position_ids"] = position_ids

            # Disable load balancing during computation
            _cp_options.enable_load_balance = False

            with sdpa_kernel(sdpa_backend):  # Context manager for attention backend
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
                    # Extract labels before forward pass
                    labels = batch.pop("labels")
                    outputs = model(**batch)  # [mbs, seq_len/cp]
                    loss = outputs.loss
                    logits = outputs.logits

                    # Compute loss using shifted label strategy
                    loss = model.loss_function(
                        logits=logits, labels=None, shift_labels=labels, vocab_size=model.config.vocab_size
                    )
                    loss.backward()

                # Sync gradients across distributed meshes
                if use_ddp:
                    pass  # FSDP/DDP handles gradient synchronization automatically
                else:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if isinstance(param.grad, DTensor):
                                local_grad = param.grad.to_local()
                                torch.distributed.all_reduce(
                                    local_grad, op=torch.distributed.ReduceOp.SUM, group=mesh.get_group()
                                )
                                local_grad = local_grad / mesh.size()
                                param.grad = DTensor.from_local(
                                    local_grad, device_mesh=param.grad.device_mesh, placements=param.grad.placements
                                )
                            else:
                                torch.distributed.all_reduce(
                                    param.grad, op=torch.distributed.ReduceOp.AVG, group=mesh.get_group()
                                )

                # Apply gradient clipping with optional foreach support
                if hasattr(model, "clip_grad_norm_"):
                    gradnorm = model.clip_grad_norm_(max_norm=1.0, norm_type=2.0)  # Standard FSDP method
                else:
                    assert len(list(model.parameters())) > 5, "No parameters found in model. Probably DDP bug.."
                    gradnorm = clip_grad_norm_
                            (model.parameters(), max_norm=1.0, norm_type=2.0, foreach=True)

                optimizer.step()

                # Average loss across distributed groups before logging
                if dist.is_initialized() and (cp_mesh.size() > 1 or dp_mesh.size() > 1):
                    dist.all_reduce(loss, group=world_mesh["dp_cp"].get_group(), op=dist.ReduceOp.AVG)
                current_loss = loss.item()

                # Log metrics and gradnorm to wandb
                if not dist.is_initialized() or dist.get_rank() == 0:
                    logger.info(
                        f"Step: {step} | GBS: {global_batch_size} | DP: {dp_mesh.size()} | TP: {tp_mesh.size()} | CP: {cp_mesh.size()} | Loss: {current_loss} | Gradnorm: {gradnorm} | lr: {LR}"
                    )
                    wandb.log(
                        {
                            "train/loss": current_loss,
                            "train/gradnorm": gradnorm,
                            "step": step,
                            "lr": LR,
                            "GBS": global_batch_size,
                        }
                    )

            step += 1  # Increment iteration counter

    logger.info("Training loop finished.")

    # Persist model checkpoint using distributed checkpointing
    if dist.is_initialized():
        state_dict = {"app": AppState(model, optimizer)}
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=CHECKPOINT_DIR,
        )
        logger.info(f"Saved checkpoint to {CHECKPOINT_DIR}")
    else:
        # Fallback to standard saving for single-process execution
        save_dir = "test_model_nondist"
        model.save_pretrained(save_dir, safe_serialization=False)
        tokenizer.save_pretrained(save_dir)  # Persist tokenizer too
        logger.info(f"Saved model to {save_dir}")

    dist.destroy_process_group()
    logger.info("Cleaned up distributed process group")
    # Close wandb run on primary rank
    if dist.get_rank() == 0:
        wandb.finish()
        logger.info("Wandb run finished.")


def _all_reduce_grads(model, world_mesh, use_ddp):
    """
    Synchronize gradient tensors across distributed process groups.
    
    Args: model (model), world_mesh (DeviceMesh), use_ddp (bool)
    
    Returns: None after completing synchronization
    """
    cp_mesh = world_mesh["cp"]
    if use_ddp:
        # DDP/FSDP handles gradient synchronization internally
        mesh = cp_mesh
    else:
        mesh = world_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
    if dist.is_initialized() and mesh.size() > 1:
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Handle cross-mesh communication with DTensor gradients
                if isinstance(param.grad, DTensor):
                    local_grad = param.grad.to_local()
                    torch.distributed.all_reduce(
                        local_grad, op=torch.distributed.ReduceOp.SUM, group=mesh.get_group()
                    )
                    local_grad = local_grad / mesh.size()
                    param.grad = DTensor.from_local(
                        local_grad, device_mesh=param.grad.device_mesh, placements=param.grad.placements
                    )
                else:
                    # Handle non-DTensor gradients directly
                    torch.distributed.all_reduce(
                        param.grad, op=torch.distributed.ReduceOp.AVG, group=mesh.get_group()
                    )


class AppState(StatefulABC):
    """
    State container for checkpointing model and optimizer parameters together.
    
    This wrapper enables distributed checkpoint saving with both 
    model weights and optimizer states preserved.
    
    Args: model (model instance), optimizer (optimizer instance)
    
    Attributes:
        model: The neural network model instance
        optimizer: The training optimizer instance
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        """Extract state dictionary from model and optimizer."""
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        """Load checkpointed state into model and optimizer."""
        set_state_dict(
            self.model, self.optimizer, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optim"]
        )


def clip_grad_norm_(
    parameters: IterableABC[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    """
    Clip the gradient norm across an iterable of model parameters.
    
    This function normalizes gradients and applies L2 normalization-based clipping
    to prevent exploding gradients during training.
    
    Args:
        parameters: Iterable of trainable parameter tensors
        max_norm: Maximum allowed gradient norm value
        norm_type: Norm type for calculation (default 2.0)
        error_if_nonfinite: Raise if NaN/Inf detected
        foreach: Use foreach implementation if True
    
    Returns:
        Scalar tensor representing the clipping coefficient
    """
    # Filter out parameters with no gradients
    parameters = [p for p in parameters if p.grad is not None]
    assert len(parameters) > 0, "No parameters with gradients found"

    # Calculate aggregate norm across all parameters
    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)

    # Convert DTensor to local tensor if needed
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    return total_norm


if __name__ == "__main__":
    __main__()