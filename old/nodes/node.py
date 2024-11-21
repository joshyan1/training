import os
import requests
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import logging

SERVER_URL = "http://10.36.39.85:11435"  # Flask server IP and port

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '10.36.39.85'  # Ensure this is an IPv4 address
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

def train(rank, world_size, master_ip):
    setup(1, master_ip)
    torch.manual_seed(0)

    # Enable MPS for Apple Silicon
    device = torch.device("mps")
    print(f"Rank {rank} is using device: {device}")

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
    for epoch in range(5):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)  # Move data to MPS device
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}, Loss: {loss.item()}")

        # Aggregate loss across all ranks
        total_loss = torch.tensor(epoch_loss).to(device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        avg_loss = total_loss.item() / world_size

        if rank == 0:
            print(f"Epoch {epoch}: Average Loss: {avg_loss}")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_id', type=str, required=True, help='Unique identifier for this worker node')
    parser.add_argument('--compute_power', type=str, required=True, help='Compute power description')
    parser.add_argument('--location', type=str, required=True, help='Location of this worker node')
    parser.add_argument('--world_size', type=int, required=True, help='Total number of nodes in the system')
    args = parser.parse_args()

    # Query the Flask server to get the master's IP
    response = requests.get(f"{SERVER_URL}/get_master_ip")
    master_ip = response.json().get("master_ip")
    if not master_ip:
        raise RuntimeError("Failed to retrieve master IP from Flask server.")
    print(f"Received master IP: {master_ip}")

    # Start training
    world_size = args.world_size
    torch.multiprocessing.spawn(train, args=(world_size, master_ip), nprocs=world_size)
