import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.multiprocessing as mp
from fsdp import launch_training, FSDPConfig, ShardingMode  # Import from fsdp.py

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.network(x.view(x.size(0), -1))

def main():
    # Set up dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Configure FSDP
    fsdp_config = FSDPConfig(
        sharding_mode=ShardingMode.FULL_SHARD,
        mixed_precision=False,  # Set to False for CPU training
        cpu_offload=False
    )
    
    # Create model
    model = SimpleModel()
    
    # Launch training
    launch_training(
        model=model,
        dataset=dataset,
        num_epochs=2,
        batch_size=64,
        world_size=2,
        fsdp_config=fsdp_config
    )

if __name__ == "__main__":
    # Initialize multiprocessing method
    mp.set_start_method('spawn', force=True)
    
    # Add freeze_support for Windows compatibility
    mp.freeze_support()
    
    # Run main function
    main()