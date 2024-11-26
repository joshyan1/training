import numpy as np
import grpc
from concurrent import futures
import torch
import torch.nn as nn
import torch.nn.functional as F
from protos import neural_network_pb2
from protos import neural_network_pb2_grpc

class NeuralLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.weights = torch.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = torch.zeros(output_size)
        self.activation = activation
        
        # Gradients
        self.weight_gradients = None
        self.bias_gradients = None
        
        # Cache for backprop
        self.input_cache = None
        self.output_cache = None
        
    def forward(self, X):
        self.input_cache = X
        Z = torch.matmul(X, self.weights) + self.biases
        
        if self.activation == 'relu':
            A = F.relu(Z)
        elif self.activation == 'softmax':
            A = F.softmax(Z, dim=1)
        else:
            A = Z
            
        self.output_cache = A
        return A
        
    def backward(self, dA):
        if self.activation == 'relu':
            dZ = dA * (self.output_cache > 0).float()
        else:  # For softmax, dA is already dZ
            dZ = dA
            
        m = self.input_cache.shape[0]
        
        self.weight_gradients = torch.matmul(self.input_cache.t(), dZ) / m
        self.bias_gradients = torch.sum(dZ, dim=0) / m
        
        return torch.matmul(dZ, self.weights.t())
        
    def update(self, learning_rate):
        self.weights -= learning_rate * self.weight_gradients
        self.biases -= learning_rate * self.bias_gradients

class NeuralNetworkServicer(neural_network_pb2_grpc.NeuralNetworkServiceServicer):
    def __init__(self):
        self.layers = []
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def Initialize(self, request, context):
        """Initialize the neural network layers"""
        try:
            self.layers = []
            for layer_config in request.layer_configs:
                layer = NeuralLayer(
                    input_size=layer_config.input_size,
                    output_size=layer_config.output_size,
                    activation=layer_config.activation
                )
                self.layers.append(layer)
                
            return neural_network_pb2.InitializeResponse(
                status="success",
                message=f"Initialized {len(self.layers)} layers"
            )
        except Exception as e:
            return neural_network_pb2.InitializeResponse(
                status="error",
                message=str(e)
            )

    def Forward(self, request, context):
        """Forward pass through the layers"""
        try:
            # Reshape input data
            input_data = torch.tensor(
                request.input, 
                dtype=torch.float32
            ).reshape(request.batch_size, request.input_size)
            
            # Forward through all layers
            A = input_data
            for layer in self.layers:
                A = layer.forward(A)
            
            # Prepare response
            return neural_network_pb2.ForwardResponse(
                output=A.flatten().tolist(),
                batch_size=A.shape[0],
                output_size=A.shape[1]
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return neural_network_pb2.ForwardResponse()

    def Backward(self, request, context):
        """Backward pass for gradient computation"""
        try:
            # Reshape gradient input
            grad_input = torch.tensor(
                request.grad_input,
                dtype=torch.float32
            ).reshape(request.batch_size, request.input_size)
            
            # Backward through all layers
            dA = grad_input
            for layer in reversed(self.layers):
                dA = layer.backward(dA)
            
            # Prepare response
            return neural_network_pb2.BackwardResponse(
                grad_output=dA.flatten().tolist(),
                batch_size=dA.shape[0],
                output_size=dA.shape[1]
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return neural_network_pb2.BackwardResponse()

    def Update(self, request, context):
        """Update parameters using computed gradients"""
        try:
            for layer in self.layers:
                layer.update(request.learning_rate)
            return neural_network_pb2.UpdateResponse(status="success")
        except Exception as e:
            return neural_network_pb2.UpdateResponse(status=str(e))

def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    neural_network_pb2_grpc.add_NeuralNetworkServiceServicer_to_server(
        NeuralNetworkServicer(), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Device server started on port {port}")
    server.wait_for_termination()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()
    serve(args.port)