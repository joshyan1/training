import zmq
import pickle
import numpy as np

class Device:
    def __init__(self, input_size, output_size, activation='relu', device_id=0):
        self.device_id = device_id
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation
        
    def quantize(self, x, bits=8):
        """Quantize gradients to reduce communication overhead"""
        max_val = np.max(np.abs(x))
        if max_val == 0:
            return x
        
        scale = (2 ** (bits - 1) - 1) / max_val
        quantized = np.round(x * scale)
        return quantized / scale
    
    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W) + self.b
        
        if self.activation == 'relu':
            self.A = np.maximum(0, self.Z)
        else:  # softmax
            exp_z = np.exp(self.Z - np.max(self.Z, axis=1, keepdims=True))
            self.A = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.A
    
    def backward(self, dA):
        m = self.A_prev.shape[0]
        
        if self.activation == 'relu':
            dZ = dA * (self.Z > 0)
        else:  # softmax
            dZ = dA
            
        # Quantize gradients
        self.dW = self.quantize(np.dot(self.A_prev.T, dZ) / m)
        self.db = self.quantize(np.sum(dZ, axis=0, keepdims=True) / m)
        dA_prev = self.quantize(np.dot(dZ, self.W.T))
        
        return dA_prev
    
    def update(self, learning_rate):
        # Apply quantized updates
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

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