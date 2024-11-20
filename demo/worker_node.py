import requests
import socket

# Flask server configuration
SERVER_IP = "CENTRAL_SERVER_IP"  # Replace with the Flask server's IP
SERVER_PORT = 5000

def register_worker(node_id, compute_resources):
    """Register this worker node with the Flask server."""
    # Get this machine's local IP address
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    
    # Send registration data to the server
    url = f"http://{SERVER_IP}:{SERVER_PORT}/register"
    payload = {
        "node_id": node_id,
        "ip_address": ip_address,
        "compute_resources": compute_resources,
    }
    
    try:
        response = requests.post(url, json=payload)
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error registering worker: {e}")

if __name__ == "__main__":
    node_id = input("Enter a unique node ID: ")
    compute_resources = {
        "num_cores": 4,  # Number of CPU cores
        "memory": "16GB",  # Amount of RAM
        "gpu": True,  # Whether this machine has a GPU
    }
    register_worker(node_id, compute_resources)