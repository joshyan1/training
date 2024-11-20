import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import requests
import logging

SERVER_URL = "http://10.36.39.85:11435"  # Flask server IP and port
RESULTS_FILE = "training_results.txt"  # File to save training results


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '10.36.39.85'  # Master node's own IP
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

MODEL_SAVE_PATH = "simple_model.pth"  # Path to save the trained model

def train(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(0)

    # Enable MPS for Apple Silicon
    device = torch.device("mps")
    logging.info(f"Rank {rank} using device: {device}")

    # Data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

    # Model and optimizer
    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    with open(RESULTS_FILE, "a") as f:
        for epoch in range(5):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)  # Move data to MPS device
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if rank == 0 and batch_idx % 20 == 0:
                    log_message = f"Epoch {epoch} Batch {batch_idx}, Loss: {loss.item()}"
                    logging.info(log_message)
                    f.write(log_message + "\n")
                    print(log_message)
    
    # Save the model only from the master process (rank 0)
    if rank == 0:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logging.info(f"Model saved to {MODEL_SAVE_PATH}")

    cleanup()

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--node_id', type=str, required=True, help='Unique identifier for this master node')
    parser.add_argument('--compute_power', type=str, required=True, help='Compute power description')
    parser.add_argument('--location', type=str, required=True, help='Location of this master node')
    parser.add_argument('--world_size', type=int, required=True, help='Total number of nodes in the system')
    args = parser.parse_args()

    # Register master node with the server
    master_ip = "http://10.36.39.85:12355"  # Replace with the actual IP of the master node
    data = {
        "node_id": args.node_id,
        "compute_power": args.compute_power,
        "location": args.location,
        "master_ip": master_ip,
    }
    response = requests.post(f"{SERVER_URL}/register", json=data)
    logging.info(f"Server response: {response.json()}")

    # Start training
    world_size = args.world_size
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
