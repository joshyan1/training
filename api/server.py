from flask import Flask, request, jsonify
import zmq
from nn.coordinator import DistributedNeuralNetwork
import threading
import os

app = Flask(__name__)

# Global state
coordinator = None
lock = threading.Lock()
registered_devices = {}

@app.route('/api/devices/register', methods=['POST'])
def register_device():
    """Register a new device with its port number"""
    data = request.get_json()
    
    if not data or 'port' not in data:
        return jsonify({'error': 'Port number is required'}), 400
        
    port = data['port']
    
    with lock:
        if port in registered_devices:
            return jsonify({'error': 'Device already registered'}), 409
            
        # Initialize coordinator if this is the first device
        global coordinator
        if coordinator is None:
            coordinator = DistributedNeuralNetwork(layer_sizes=[784, 128, 64, 10])
            
        # Try to connect to the device
        if coordinator.connect_to_device(port):
            device_id = len(registered_devices) + 1
            registered_devices[port] = device_id
            return jsonify({
                'message': 'Device registered successfully',
                'device_id': device_id
            }), 201
        else:
            return jsonify({'error': 'Failed to connect to device'}), 500

@app.route('/api/devices', methods=['GET'])
def get_devices():
    """Get list of registered devices"""
    return jsonify({
        'devices': [
            {'port': port, 'device_id': device_id} 
            for port, device_id in registered_devices.items()
        ],
        'max_devices': coordinator.max_devices if coordinator else 0,
        'connected_devices': len(registered_devices)
    })

@app.route('/api/network/initialize', methods=['POST'])
def initialize_network():
    """Initialize the neural network across registered devices"""
    if not coordinator:
        return jsonify({'error': 'No devices registered'}), 400
        
    if len(registered_devices) == 0:
        return jsonify({'error': 'No devices available'}), 400
        
    try:
        coordinator.initialize_devices()
        return jsonify({'message': 'Network initialized successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to initialize network: {str(e)}'}), 500

@app.route('/api/network/train', methods=['POST'])
def start_training():
    """Start the training process"""
    if not coordinator:
        return jsonify({'error': 'Network not initialized'}), 400
        
    data = request.get_json() or {}
    epochs = data.get('epochs', 10)
    learning_rate = data.get('learning_rate', 0.1)
    
    try:
        # Start training in a separate thread
        def train_thread():
            from torch.utils.data import DataLoader
            from torchvision import datasets, transforms
            
            # Define transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # Load MNIST dataset
            train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
            val_dataset = datasets.MNIST('data', train=False, transform=transform)
            
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1000)
            
            coordinator.train(train_loader, val_loader, epochs=epochs, learning_rate=learning_rate)
        
        thread = threading.Thread(target=train_thread)
        thread.start()
        
        return jsonify({
            'message': 'Training started',
            'epochs': epochs,
            'learning_rate': learning_rate
        })
    except Exception as e:
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500

@app.route('/api/devices/<int:port>', methods=['DELETE'])
def unregister_device(port):
    """Unregister a device"""
    if port not in registered_devices:
        return jsonify({'error': 'Device not found'}), 404
        
    with lock:
        del registered_devices[port]
        if len(registered_devices) == 0:
            global coordinator
            coordinator = None
            
        return jsonify({'message': 'Device unregistered successfully'})

def create_app():
    """Create and configure the Flask app"""
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=4000, debug=True)