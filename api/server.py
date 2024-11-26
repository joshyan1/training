from flask import Flask, request, jsonify
import threading
from nn.coordinator import DistributedNeuralNetwork
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

app = Flask(__name__)

# Global state
neural_network = None
training_thread = None
connected_ports = []
layer_sizes = [784, 128, 64, 10]

def load_data():
    """Load and prepare MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000)
    
    return train_loader, val_loader

def train_network(nn, train_loader, val_loader):
    """Training function to run in separate thread"""
    try:
        nn.initialize_devices()
        nn.train(train_loader, val_loader)
    except Exception as e:
        print(f"Training error: {str(e)}")

@app.route('/api/register_device', methods=['POST'])
def register_device():
    """Register a new device with port number"""
    global neural_network, connected_ports
    
    data = request.get_json()
    if not data or 'port' not in data:
        return jsonify({'error': 'Port number required'}), 400
    
    port = data['port']
    
    # Initialize neural network if not exists
    if neural_network is None:
        neural_network = DistributedNeuralNetwork(layer_sizes)
    
    # Check if we've reached max devices
    if len(connected_ports) >= neural_network.max_devices:
        return jsonify({
            'error': f'Maximum number of devices ({neural_network.max_devices}) reached'
        }), 400
    
    # Try to connect to the device
    if neural_network.connect_to_device(port):
        connected_ports.append(port)
        return jsonify({
            'status': 'success',
            'message': f'Device registered on port {port}',
            'connected_devices': len(connected_ports),
            'max_devices': neural_network.max_devices
        })
    else:
        return jsonify({
            'error': f'Failed to connect to device on port {port}'
        }), 500

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Initialize and start the training process"""
    global neural_network, training_thread, connected_ports
    
    if not connected_ports:
        return jsonify({'error': 'No devices connected'}), 400
        
    if training_thread and training_thread.is_alive():
        return jsonify({'error': 'Training already in progress'}), 400
    
    try:
        # Verify devices are still connected
        dead_ports = []
        for port in connected_ports:
            if not neural_network.connect_to_device(port):
                dead_ports.append(port)
        
        # Remove dead ports
        for port in dead_ports:
            connected_ports.remove(port)
            
        if not connected_ports:
            return jsonify({'error': 'All devices are disconnected. Please reconnect devices.'}), 400
        
        # Load data
        train_loader, val_loader = load_data()
        
        # Start training in a separate thread
        training_thread = threading.Thread(
            target=train_network,
            args=(neural_network, train_loader, val_loader)
        )
        training_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Training started',
            'connected_devices': len(connected_ports)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current training status"""
    global neural_network, training_thread, connected_ports
    
    status = {
        'connected_devices': len(connected_ports),
        'max_devices': neural_network.max_devices if neural_network else None,
        'training_in_progress': training_thread.is_alive() if training_thread else False
    }
    
    return jsonify(status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)