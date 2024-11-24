import zmq
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

class DistributedNeuralNetwork:
    def __init__(self, layer_sizes, quantization_bits=8):
        self.layer_sizes = layer_sizes
        self.device_connections = {}
        self.required_devices = len(layer_sizes) - 1
        self.quantization_bits = quantization_bits
        self.context = zmq.Context()
        # Set device to MPS if available, else CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def quantize(self, x):
        """Quantize data before sending"""
        # Convert torch tensor to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            
        max_val = np.max(np.abs(x))
        if max_val == 0:
            return x
        
        scale = (2 ** (self.quantization_bits - 1) - 1) / max_val
        quantized = np.round(x * scale)
        return quantized / scale

    def connect_to_device(self, port):
        """Connect to a device on the specified port"""
        try:
            socket = self.context.socket(zmq.REQ)  # Request socket
            socket.connect(f"tcp://localhost:{port}")
            device_id = len(self.device_connections) + 1
            self.device_connections[device_id] = socket
            print(f"Connected to device {device_id} at port {port}")
            return True
        except Exception as e:
            print(f"Failed to connect to device at port {port}: {e}")
            return False

    def initialize_devices(self):
        """Initialize all connected devices with their layer configurations"""
        layers_per_device = len(self.layer_sizes) // len(self.device_connections)
        extra_layers = len(self.layer_sizes) % len(self.device_connections)
        
        current_layer = 0
        for device_id, socket in self.device_connections.items():
            # Calculate number of layers for this device
            n_layers = layers_per_device + (1 if device_id <= extra_layers else 0)
            
            # Create layer configurations for this device
            layer_configs = []
            for i in range(n_layers):
                if current_layer < len(self.layer_sizes) - 1:
                    layer_configs.append({
                        'input_size': self.layer_sizes[current_layer],
                        'output_size': self.layer_sizes[current_layer + 1],
                        'activation': 'relu' if current_layer < len(self.layer_sizes) - 2 else 'softmax'
                    })
                    current_layer += 1
            
            # Initialize device with its layer configurations
            socket.send_pyobj({
                'command': 'init',
                'layer_configs': layer_configs,
                'device_id': device_id
            })
            response = socket.recv_pyobj()
            print(f"Initialized device {device_id} with {len(layer_configs)} layers: {response}")

    def forward(self, X):
        """Distributed forward pass with quantized data"""
        # Convert torch tensor to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            
        # Ensure input is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        elif len(X.shape) == 3:  # For MNIST images (batch_size, 28, 28)
            X = X.reshape(X.shape[0], -1)
        elif len(X.shape) == 4:  # For MNIST images with channel (batch_size, 1, 28, 28)
            X = X.reshape(X.shape[0], -1)
        
        A = self.quantize(X)
        activations = [X]
        
        for device_id in range(1, self.required_devices + 1):
            socket = self.device_connections[device_id]
            socket.send_pyobj({
                'command': 'forward',
                'input': A
            })
            response = socket.recv_pyobj()
            A = response['output']
            activations.append(A)
            
        return activations

    def backward(self, activations, y_true):
        """Distributed backward pass with quantized gradients"""
        m = y_true.shape[0]
        y_onehot = np.zeros_like(activations[-1])
        y_onehot[np.arange(m), y_true] = 1
        dA = self.quantize(activations[-1] - y_onehot)
        
        for device_id in reversed(range(1, self.required_devices + 1)):
            socket = self.device_connections[device_id]
            socket.send_pyobj({
                'command': 'backward',
                'grad_input': dA
            })
            response = socket.recv_pyobj()
            dA = response['grad_output']

    def update_parameters(self, learning_rate):
        """Update parameters on all devices"""
        for device_id in range(1, self.required_devices + 1):
            socket = self.device_connections[device_id]
            socket.send_pyobj({
                'command': 'update',
                'learning_rate': learning_rate
            })
            socket.recv_pyobj()  # Wait for acknowledgment

    def train(self, train_loader, val_loader, epochs=100, learning_rate=0.1):
        print("\nStarting distributed training across devices...")
        print(f"Learning rate: {learning_rate}")
        print(f"Quantization bits: {self.quantization_bits}")
        print(f"Device: {self.device}")
        print("-" * 100)
        
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            n_batches = 0
            
            # Training loop
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass with quantized data
                activations = self.forward(data)
                y_pred = activations[-1]
                
                batch_loss = self.compute_loss(target.cpu().numpy(), y_pred)
                batch_acc = self.compute_accuracy(target.cpu().numpy(), y_pred)
                
                epoch_loss += batch_loss
                epoch_acc += batch_acc
                n_batches += 1
                
                # Backward pass with quantized gradients
                self.backward(activations, target.cpu().numpy())
                self.update_parameters(learning_rate)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"\rEpoch {epoch+1}/{epochs} "
                          f"[Batch {batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {batch_loss:.4f} "
                          f"Acc: {batch_acc:.4f}", end="")
            
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            
            # Validation loop
            val_loss = 0
            val_acc = 0
            n_val_batches = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    val_activations = self.forward(data)
                    val_pred = val_activations[-1]
                    val_loss += self.compute_loss(target.cpu().numpy(), val_pred)
                    val_acc += self.compute_accuracy(target.cpu().numpy(), val_pred)
                    n_val_batches += 1
            
            val_loss /= n_val_batches
            val_acc /= n_val_batches
            
            print(f"\nEpoch {epoch+1:2d}/{epochs} - "
                  f"Loss: {epoch_loss:.4f} - "
                  f"Acc: {epoch_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val Acc: {val_acc:.4f}")
            print("-" * 100)

    def compute_loss(self, y_true, y_pred):
        """Compute cross entropy loss"""
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
        y_onehot = np.zeros_like(y_pred)
        y_onehot[np.arange(m), y_true] = 1
        return -np.sum(y_onehot * np.log(y_pred)) / m

    def compute_accuracy(self, y_true, y_pred):
        """Compute accuracy from predictions"""
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true)

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset using PyTorch
    print("\nLoading MNIST dataset...")
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Ensure proper reshaping of data
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000)
    
    layer_sizes = [784, 128, 64, 10]  # First layer must match flattened input size
    nn = DistributedNeuralNetwork(layer_sizes)
    
    # Connect to devices
    base_ports = [5001, 5002, 5003]
    for port in base_ports:
        nn.connect_to_device(port)
    
    if len(nn.device_connections) < nn.required_devices:
        print(f"Not enough devices connected. Need {nn.required_devices}, found {len(nn.device_connections)}")
        return
    
    response = input("\nStart training? (y/n): ").lower()
    if response != 'y':
        print("Training cancelled.")
        return
    
    nn.initialize_devices()
    
    nn.train(train_loader, val_loader)

if __name__ == "__main__":
    main() 