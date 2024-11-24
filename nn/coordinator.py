import zmq
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DistributedNeuralNetwork:
    def __init__(self, layer_sizes, quantization_bits=8):
        self.layer_sizes = layer_sizes
        self.device_connections = {}
        self.required_devices = len(layer_sizes) - 1
        self.quantization_bits = quantization_bits
        self.context = zmq.Context()
        
    def quantize(self, x):
        """Quantize data before sending"""
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
        """Initialize all connected devices"""
        for device_id, socket in self.device_connections.items():
            i = device_id - 1
            socket.send_pyobj({
                'command': 'init',
                'input_size': self.layer_sizes[i],
                'output_size': self.layer_sizes[i + 1],
                'activation': 'relu' if i < self.required_devices - 1 else 'softmax',
                'device_id': device_id
            })
            response = socket.recv_pyobj()
            print(f"Initialized device {device_id}: {response}")

    def forward(self, X):
        """Distributed forward pass with quantized data"""
        A = self.quantize(X)  # Quantize input
        activations = [X]  # Keep original input for backprop
        
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

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=256, learning_rate=0.1):
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Define validation batch parameters
        val_batch_size = 1000
        n_val_batches = (len(X_val) + val_batch_size - 1) // val_batch_size
        
        print("\nStarting distributed training across devices...")
        print(f"Training samples: {n_samples}, Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        print(f"Quantization bits: {self.quantization_bits}")
        print("-" * 100)
        
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            epoch_acc = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass with quantized data
                activations = self.forward(X_batch)
                y_pred = activations[-1]
                
                batch_loss = self.compute_loss(y_batch, y_pred)
                batch_acc = self.compute_accuracy(y_batch, y_pred)
                
                epoch_loss += batch_loss
                epoch_acc += batch_acc
                
                # Backward pass with quantized gradients
                self.backward(activations, y_batch)
                self.update_parameters(learning_rate)
                
                if (i + 1) % 10 == 0:
                    print(f"\rEpoch {epoch+1}/{epochs} "
                          f"[Batch {i+1}/{n_batches}] "
                          f"Loss: {batch_loss:.4f} "
                          f"Acc: {batch_acc:.4f}", end="")
            
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            
            # Evaluate validation in chunks
            val_loss = 0
            val_acc = 0
            for i in range(n_val_batches):
                start_idx = i * val_batch_size
                end_idx = min((i + 1) * val_batch_size, len(X_val))
                
                X_val_batch = X_val[start_idx:end_idx]
                y_val_batch = y_val[start_idx:end_idx]
                
                val_activations = self.forward(X_val_batch)
                val_pred = val_activations[-1]
                val_loss += self.compute_loss(y_val_batch, val_pred) * len(X_val_batch)
                val_acc += self.compute_accuracy(y_val_batch, val_pred) * len(X_val_batch)
            
            val_loss /= len(X_val)
            val_acc /= len(X_val)
            
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
    # Initialize network
    layer_sizes = [784, 128, 64, 10]
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
    
    print("\nLoading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype('float32') / 255.
    y = mnist.target.astype('int32')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    nn.train(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    main() 