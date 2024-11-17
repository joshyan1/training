import os
import requests
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

SERVER_URL = "http://<FLASK_SERVER_IP>:5000"  # Replace <FLASK_SERVER_IP> with the Flask server's IP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '<FLASK_SERVER_IP>'  # Replace with Flask server's IP
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
    model = FSDP(model)
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
    parser.add_argument('--node_id', type=str, required=True, help='Unique identifier for this node')
    parser.add_argument('--compute_power', type=str, required=True, help='Compute power description')
    parser.add_argument('--location', type=str, required=True, help='Location of this node')
    parser.add_argument('--world_size', type=int, required=True, help='Total number of nodes in the system')
    args = parser.parse_args()

    # Register the node with the server
    data = {
        "node_id": args.node_id,
        "compute_power": args.compute_power,
        "location": args.location,
    }
    response = requests.post(f"{SERVER_URL}/register", json=data)
    print(response.json())

    # Start training
    world_size = args.world_size
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
