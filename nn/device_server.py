import zmq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Device:
    def __init__(self, input_size, output_size, activation='relu', device_id=0):
        self.device_id = device_id
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize weights using PyTorch
        self.W = torch.randn(input_size, output_size, device=self.device) * np.sqrt(2.0 / input_size)
        self.b = torch.zeros(1, output_size, device=self.device)
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)
        
        self.activation = activation
        
    def quantize(self, x, bits=8):
        """Quantize gradients to reduce communication overhead"""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            
        max_val = np.max(np.abs(x))
        if max_val == 0:
            return x
        
        scale = (2 ** (bits - 1) - 1) / max_val
        quantized = np.round(x * scale)
        return quantized / scale
    
    def forward(self, A_prev):
        """Forward pass with proper dimension handling"""
        # Convert numpy array to torch tensor if needed
        if isinstance(A_prev, np.ndarray):
            # Ensure input is 2D
            if len(A_prev.shape) == 1:
                A_prev = A_prev.reshape(1, -1)
            elif len(A_prev.shape) == 3:  # For MNIST images
                A_prev = A_prev.reshape(A_prev.shape[0], -1)
            elif len(A_prev.shape) == 4:  # For MNIST images with channel
                A_prev = A_prev.reshape(A_prev.shape[0], -1)
            
            A_prev = torch.from_numpy(A_prev).float().to(self.device)
        
        # Ensure tensor is 2D
        if len(A_prev.shape) == 1:
            A_prev = A_prev.unsqueeze(0)
        elif len(A_prev.shape) > 2:
            A_prev = A_prev.view(A_prev.size(0), -1)
            
        self.A_prev = A_prev
        
        # Debug prints
        print(f"A_prev shape: {A_prev.shape}")
        print(f"W shape: {self.W.shape}")
        print(f"b shape: {self.b.shape}")
        
        self.Z = torch.mm(A_prev, self.W) + self.b
        
        if self.activation == 'relu':
            self.A = F.relu(self.Z)
        else:  # softmax
            self.A = F.softmax(self.Z, dim=1)
            
        return self.A.detach().cpu().numpy()
    
    def backward(self, dA):
        """Backward pass with proper dimension handling"""
        # Convert numpy array to torch tensor
        if isinstance(dA, np.ndarray):
            # Ensure input is 2D
            if len(dA.shape) == 1:
                dA = dA.reshape(1, -1)
            elif len(dA.shape) > 2:
                dA = dA.reshape(dA.shape[0], -1)
            
            dA = torch.from_numpy(dA).float().to(self.device)
        
        # Ensure tensor is 2D
        if len(dA.shape) == 1:
            dA = dA.unsqueeze(0)
        elif len(dA.shape) > 2:
            dA = dA.view(dA.size(0), -1)
            
        m = self.A_prev.size(0)
        
        if self.activation == 'relu':
            dZ = dA * (self.Z > 0)
        else:  # softmax
            dZ = dA
            
        # Compute gradients
        self.dW = self.quantize(torch.mm(self.A_prev.t(), dZ).detach().cpu().numpy() / m)
        self.db = self.quantize(torch.sum(dZ, dim=0, keepdim=True).detach().cpu().numpy() / m)
        dA_prev = self.quantize(torch.mm(dZ, self.W.t()).detach().cpu().numpy())
        
        return dA_prev
    
    def update(self, learning_rate):
        # Convert numpy gradients to torch tensors
        dW = torch.from_numpy(self.dW).float().to(self.device)
        db = torch.from_numpy(self.db).float().to(self.device)
        
        # Apply updates
        with torch.no_grad():
            self.W -= learning_rate * dW
            self.b -= learning_rate * db

class DeviceServer:
    def __init__(self, port):
        self.port = port
        self.device = None
        
        # Setup ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)  # Reply socket
        self.socket.bind(f"tcp://*:{port}")
        
    def start(self):
        print(f"Starting device server on port {self.port}")
        
        while True:
            # Wait for message
            message = self.socket.recv_pyobj()  # Automatically unpickles
            command = message['command']
            
            if command == 'init':
                self.device = Device(
                    input_size=message['input_size'],
                    output_size=message['output_size'],
                    activation=message['activation'],
                    device_id=message['device_id']
                )
                self.socket.send_pyobj({
                    'status': 'initialized',
                    'device_id': self.device.device_id
                })
                
            elif command == 'forward':
                A_prev = message['input']
                output = self.device.forward(A_prev)
                self.socket.send_pyobj({'output': output})
                
            elif command == 'backward':
                dA = message['grad_input']
                dA_prev = self.device.backward(dA)
                self.socket.send_pyobj({'grad_output': dA_prev})
                
            elif command == 'update':
                self.device.update(message['learning_rate'])
                self.socket.send_pyobj({'status': 'updated'})

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python device_server.py <port>")
        sys.exit(1)
        
    port = int(sys.argv[1])
    device_server = DeviceServer(port)
    device_server.start()