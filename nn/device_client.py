import subprocess
import requests
import sys
import time
import signal
import os

def get_local_ip():
    """Get the local IP address of this machine"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to an external server (doesn't actually send anything)
        s.connect(('8.8.8.8', 80))
        IP = s.getsockname()[0]
    except Exception:
        # Fallback to localhost if unable to determine IP
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

class DeviceClient:
    def __init__(self, api_url="http://localhost:4000", device_port=None):
        self.api_url = api_url
        self.device_port = device_port or self._find_available_port(start_port=5001)
        self.device_process = None
        self.local_ip = get_local_ip()
        
    def _find_available_port(self, start_port):
        """Find an available port starting from start_port"""
        import socket
        
        port = start_port
        while port < start_port + 100:  # Try up to 100 ports
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # Bind to all interfaces instead of just localhost
                sock.bind(('0.0.0.0', port))
                sock.close()
                return port
            except OSError:
                port += 1
            finally:
                sock.close()
        raise RuntimeError("Could not find an available port")

    def start_device_server(self):
        """Start the device server as a subprocess"""
        try:
            # Get the absolute path to device_server.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            server_path = os.path.join(current_dir, 'device_server.py')
            
            # Start the device server
            self.device_process = subprocess.Popen(
                [sys.executable, server_path, str(self.device_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for the server to start
            time.sleep(2)
            
            if self.device_process.poll() is not None:
                # Process has terminated
                stdout, stderr = self.device_process.communicate()
                raise RuntimeError(f"Device server failed to start: {stderr.decode()}")
                
            print(f"Device server started on port {self.device_port}")
            return True
            
        except Exception as e:
            print(f"Error starting device server: {e}")
            return False

    def register_with_api(self):
        """Register this device with the API server"""
        try:
            response = requests.post(
                f"{self.api_url}/api/devices/register",
                json={
                    'port': self.device_port,
                    'ip': self.local_ip
                }
            )
            
            if response.status_code == 201:
                print("Successfully registered with API server")
                return response.json()['device_id']
            else:
                print(f"Failed to register: {response.json()['error']}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to API server: {e}")
            return None

    def cleanup(self):
        """Clean up resources and stop the device server"""
        if self.device_process:
            try:
                # Try to unregister from the API
                requests.delete(f"{self.api_url}/api/devices/{self.device_port}")
            except:
                pass
                
            # Kill the device server process
            self.device_process.terminate()
            try:
                self.device_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.device_process.kill()
            
            self.device_process = None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Start a neural network device server and register with API')
    parser.add_argument('--api-url', default='http://localhost:4000', help='API server URL')
    parser.add_argument('--port', type=int, help='Port for device server (optional)')
    args = parser.parse_args()
    
    client = DeviceClient(api_url=args.api_url, device_port=args.port)
    
    def signal_handler(signum, frame):
        print("\nReceived shutdown signal. Cleaning up...")
        client.cleanup()
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if client.start_device_server():
            device_id = client.register_with_api()
            if device_id:
                print(f"Device {device_id} is running. Press Ctrl+C to stop.")
                # Keep the main thread alive
                while True:
                    time.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.cleanup()

if __name__ == "__main__":
    main() 