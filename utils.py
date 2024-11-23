import subprocess
import json
import time
import socket

def start_ngrok(port):
    """Start an Ngrok tunnel for a given port and return the public address."""
    process = subprocess.Popen(
        ['ngrok', 'tcp', str(port), '--log=stdout'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for Ngrok to initialize and fetch the public address
    print(process.stdout)
    for line in process.stdout:
        if "url=" in line:  # Search for the Ngrok public URL
            public_url = line.split("url=")[1].strip()
            print(f"Ngrok tunnel started: {public_url}")
            return process, public_url
    
    raise Exception("Failed to start Ngrok")

def stop_ngrok(process):
    """Stop the Ngrok process."""
    process.terminate()
    print("Ngrok tunnel stopped")


def get_local_ip():
    """Fetches the local IP address of the machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Connect to an external server to get the local IP
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"  # Fallback to localhost
        
def start_localtunnel(port):
    """Start LocalTunnel for a given port and return the public URL."""
    try:
        # Start LocalTunnel process
        process = subprocess.Popen(
            ['lt', '--port', str(port), '--tcp'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for LocalTunnel to provide the URL
        for line in process.stdout:
            if "your url is:" in line:
                public_url = line.split("your url is:")[1].strip()
                print(f"LocalTunnel started: {public_url}")
                return process, public_url
        
        raise Exception("Failed to start LocalTunnel or retrieve public URL")
    except Exception as e:
        print(f"Error starting LocalTunnel: {e}")
        raise
