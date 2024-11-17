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

SERVER_URL = "http://127.0.0.1"  # Local Flask server URL
RESULTS_FILE = "training_results.txt"  # File to save training results


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Master node's own IP
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


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

                if rank == 0 and batch_idx % 10 == 0:
                    log_message = f"Epoch {epoch} Batch {batch_idx}, Loss: {loss.item()}"
                    logging.info(log_message)
                    f.write(log_message + "\n")
    
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
    master_ip = "http://10.36.39.85:11435"  # Replace with the actual IP of the master node
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
