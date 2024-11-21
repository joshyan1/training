import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ShardingMode(Enum):
    FULL_SHARD = "FULL_SHARD"
    GRAD_SHARD = "GRAD_SHARD"
    NO_SHARD = "NO_SHARD"

@dataclass
class FSDPConfig:
    """Configuration for FSDP training"""
    sharding_mode: ShardingMode = ShardingMode.FULL_SHARD
    min_num_params: int = 1e5  # Minimum number of parameters for auto wrapping
    cpu_offload: bool = False
    mixed_precision: bool = True
    backward_prefetch: bool = True
    
    def get_sharding_strategy(self) -> ShardingStrategy:
        return {
            ShardingMode.FULL_SHARD: ShardingStrategy.FULL_SHARD,
            ShardingMode.GRAD_SHARD: ShardingStrategy.SHARD_GRAD_OP,
            ShardingMode.NO_SHARD: ShardingStrategy.NO_SHARD
        }[self.sharding_mode]
    
    def get_mixed_precision_config(self):
        if not self.mixed_precision:
            return None
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
    
    def get_cpu_offload_config(self):
        if not self.cpu_offload:
            return None
        return CPUOffload(offload_params=True)

def get_auto_wrap_policy(min_num_params):
    """Create auto wrap policy with the specified minimum number of parameters"""
    def policy(module, recurse, unwrapped_params):
        return size_based_auto_wrap_policy(
            min_num_params=min_num_params,
            recurse=recurse,
            nonwrapped_numel=unwrapped_params
        )
    return policy

def run_training(rank, world_size, model, dataset, num_epochs, batch_size, fsdp_config):
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # Initialize FSDP wrapped model
    model = FSDP(
        model.to(device),
        sharding_strategy=fsdp_config.get_sharding_strategy(),
        auto_wrap_policy=get_auto_wrap_policy(fsdp_config.min_num_params),
        mixed_precision=fsdp_config.get_mixed_precision_config(),
        cpu_offload=fsdp_config.get_cpu_offload_config(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE if fsdp_config.backward_prefetch else None,
    )

    # Prepare data
    train_sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    # Training setup
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    dist.destroy_process_group()

def launch_training(model, dataset, num_epochs=10, batch_size=32, world_size=2, fsdp_config=None):
    """
    Launch distributed training with FSDP
    
    Args:
        model: The model to train
        dataset: Training dataset
        num_epochs: Number of training epochs
        batch_size: Batch size per GPU
        world_size: Number of GPUs to use
        fsdp_config: FSDP configuration, if None uses default config
    """
    if fsdp_config is None:
        fsdp_config = FSDPConfig()
    
    mp.spawn(
        run_training,
        args=(world_size, model, dataset, num_epochs, batch_size, fsdp_config),
        nprocs=world_size,
        join=True
    )