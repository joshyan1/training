import numpy as np
import torch
import grpc
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from protos import neural_network_pb2
from protos import neural_network_pb2_grpc

class DistributedNeuralNetwork:
    def __init__(self, layer_sizes, quantization_bits=8):
        self.layer_sizes = layer_sizes
        self.max_devices = len(layer_sizes) - 1  # Maximum number of devices = number of layers
        self.device_connections = {}
        self.quantization_bits = quantization_bits
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Maximum number of devices: {self.max_devices}")
        
    def quantize(self, x):
        """Quantize data before sending"""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            
        max_val = np.max(np.abs(x))
        if max_val == 0:
            return x
        
        scale = (2 ** (self.quantization_bits - 1) - 1) / max_val
        quantized = np.round(x * scale)
        return quantized / scale

    def connect_to_device(self, port):
        """Connect to a device on the specified port if under max devices"""
        if len(self.device_connections) >= self.max_devices:
            print(f"Maximum number of devices ({self.max_devices}) reached. Skipping connection to port {port}")
            return False
            
        try:
            channel = grpc.insecure_channel(f'localhost:{port}')
            # Add timeout to connection attempt
            try:
                grpc.channel_ready_future(channel).result(timeout=5)
            except grpc.FutureTimeoutError:
                print(f"Timeout while connecting to port {port}")
                return False
                
            stub = neural_network_pb2_grpc.NeuralNetworkServiceStub(channel)
            device_id = len(self.device_connections) + 1
            self.device_connections[device_id] = stub
            print(f"Connected to device {device_id} at port {port} ({len(self.device_connections)}/{self.max_devices} devices)")
            return True
        except Exception as e:
            print(f"Failed to connect to device at port {port}: {e}")
            return False

    def initialize_devices(self):
        """Initialize devices with balanced layer distribution"""
        num_devices = len(self.device_connections)
        if num_devices == 0:
            raise Exception("No devices connected. Please connect devices before initializing.")
        
        # Verify connections are alive
        dead_devices = []
        for device_id, stub in self.device_connections.items():
            try:
                # Try a simple RPC call to check connection
                request = neural_network_pb2.InitializeRequest(
                    layer_configs=[],
                    device_id=device_id
                )
                stub.Initialize(request)
            except Exception as e:
                print(f"Device {device_id} appears to be dead: {e}")
                dead_devices.append(device_id)
        
        # Remove dead devices
        for device_id in dead_devices:
            del self.device_connections[device_id]
        
        if len(self.device_connections) == 0:
            raise Exception("All devices are dead. Please reconnect devices.")
        
        num_devices = len(self.device_connections)
        num_layers = len(self.layer_sizes) - 1
        
        # Calculate layers per device
        layers_per_device = num_layers // num_devices
        extra_layers = num_layers % num_devices
        
        print(f"\nDistributing {num_layers} layers across {num_devices} devices")
        print(f"Base layers per device: {layers_per_device}")
        print(f"Extra layers to distribute: {extra_layers}")
        
        current_layer = 0
        device_layer_map = {}
        
        # Distribute layers to devices
        for device_id in sorted(self.device_connections.keys()):
            # Calculate number of layers for this device
            device_layers = layers_per_device + (1 if extra_layers > 0 else 0)
            extra_layers = max(0, extra_layers - 1)
            
            layer_configs = []
            layer_indices = []
            
            # Add layers for this device
            for _ in range(device_layers):
                if current_layer < num_layers:
                    layer_config = neural_network_pb2.LayerConfig(
                        input_size=self.layer_sizes[current_layer],
                        output_size=self.layer_sizes[current_layer + 1],
                        activation='relu' if current_layer < num_layers - 1 else 'softmax'
                    )
                    layer_configs.append(layer_config)
                    layer_indices.append(current_layer)
                    current_layer += 1
            
            device_layer_map[device_id] = layer_indices
            
            # Initialize device with its layers
            request = neural_network_pb2.InitializeRequest(
                layer_configs=layer_configs,
                device_id=device_id
            )
            response = stub.Initialize(request)
            print(f"Initialized device {device_id} with layers {layer_indices}: {response.message}")
        
        self.device_layer_map = device_layer_map
        print("\nLayer distribution complete")

    def forward(self, X):
        """Distributed forward pass with quantized data"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            
        # Ensure input is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        elif len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        elif len(X.shape) == 4:
            X = X.reshape(X.shape[0], -1)
        
        A = self.quantize(X)
        activations = [X]
        
        # Forward through each device in order
        for device_id in sorted(self.device_connections.keys()):
            stub = self.device_connections[device_id]
            request = neural_network_pb2.ForwardRequest(
                input=A.flatten().tolist(),
                batch_size=A.shape[0],
                input_size=A.shape[1]
            )
            response = stub.Forward(request)
            A = np.array(response.output).reshape(response.batch_size, response.output_size)
            activations.append(A)
            
        return activations

    def backward(self, activations, y_true):
        """Distributed backward pass with quantized gradients"""
        m = y_true.shape[0]
        y_onehot = np.zeros_like(activations[-1])
        y_onehot[np.arange(m), y_true] = 1
        dA = self.quantize(activations[-1] - y_onehot)
        
        # Backward through each device in reverse order
        for device_id in sorted(self.device_connections.keys(), reverse=True):
            stub = self.device_connections[device_id]
            request = neural_network_pb2.BackwardRequest(
                grad_input=dA.flatten().tolist(),
                batch_size=dA.shape[0],
                input_size=dA.shape[1]
            )
            response = stub.Backward(request)
            dA = np.array(response.grad_output).reshape(response.batch_size, response.output_size)

    def update_parameters(self, learning_rate):
        """Update parameters on all devices"""
        for device_id in sorted(self.device_connections.keys()):
            stub = self.device_connections[device_id]
            request = neural_network_pb2.UpdateRequest(learning_rate=learning_rate)
            stub.Update(request)

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

# Remove the main() function and its call at the bottom of the file
# Delete or comment out these lines:
# if __name__ == "__main__":
#     main() 