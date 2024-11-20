import requests
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.multiprocessing as mp

# Flask server configuration
SERVER_IP = "CENTRAL_SERVER_IP"  # Replace with the Flask server's IP
SERVER_PORT = 5000

def fetch_registered_nodes():
    """Fetch the list of registered nodes from the Flask server."""
    url = f"http://{SERVER_IP}:{SERVER_PORT}/get_nodes"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching nodes: {e}")
        return {}

def train(rank, world_size, master_ip):
    """Distributed training function."""
    # Initialize the process group
    dist.init_process_group(
        backend="gloo",  # Use "nccl" if GPUs are available
        init_method=f"tcp://{master_ip}:12345",  # Master node IP and port
        rank=rank,
        world_size=world_size,
    )

    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Create model and wrap it in DDP
    model = torch.nn.Linear(10, 1).to(device)
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    # Dummy data
    data = torch.randn(20, 10).to(device)
    target = torch.randn(20, 1).to(device)

    # Training loop
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    for epoch in range(5):
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    # Fetch registered nodes from the Flask server
    nodes = fetch_registered_nodes()
    print(f"Registered Nodes: {nodes}")

    if not nodes:
        print("No nodes available for training. Exiting.")
        exit()

    # Determine the master IP and world size
    MASTER_IP = nodes.get("master", None)
    if not MASTER_IP:
        print("No master node registered. Exiting.")
        exit()

    WORLD_SIZE = len(nodes)

    # Spawn processes for distributed training
    mp.spawn(train, args=(WORLD_SIZE, MASTER_IP), nprocs=WORLD_SIZE, join=True)