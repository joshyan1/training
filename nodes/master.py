import os
import requests
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

SERVER_URL = "http://127.0.0.1:11435"  # Flask server IP and port

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Master node's own IP
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(0)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

    model = SimpleModel().to(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}, Loss: {loss.item()}")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_id', type=str, required=True, help='Unique identifier for this master node')
    parser.add_argument('--compute_power', type=str, required=True, help='Compute power description')
    parser.add_argument('--location', type=str, required=True, help='Location of this master node')
    parser.add_argument('--world_size', type=int, required=True, help='Total number of nodes in the system')
    args = parser.parse_args()

    # Register master node with the server
    master_ip = "127.0.0.1"  # Replace with the actual IP of the master node
    data = {
        "node_id": args.node_id,
        "compute_power": args.compute_power,
        "location": args.location,
        "master_ip": master_ip,
    }
    response = requests.post(f"{SERVER_URL}/register", json=data)
    print(response.json())

    # Start training
    world_size = args.world_size
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
