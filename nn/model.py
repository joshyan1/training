import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys
import time

print("Imported necessary libraries")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Loss function
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-12, 1. - 1e-12)
    log_likelihood = -np.log(y_pred_clipped[range(m), y_true])
    return np.sum(log_likelihood) / m

class Device:
    def __init__(self, input_size, output_size, activation='relu', device_id=0):
        self.device_id = device_id
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, A_prev, training=True):
        """Forward pass on this device"""
        self.Z = np.dot(A_prev, self.W) + self.b
        if self.activation == 'relu':
            self.A = relu(self.Z)
        elif self.activation == 'softmax':
            self.A = softmax(self.Z)
        return self.A

    def backward(self, dA, A_prev):
        """Backward pass on this device"""
        m = A_prev.shape[0]
        if self.activation == 'relu':
            dZ = dA * relu_derivative(self.Z)
        elif self.activation == 'softmax':
            dZ = dA
        
        self.dW = np.dot(A_prev.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        self.dA_prev = np.dot(dZ, self.W.T)
        return self.dA_prev

    def update_parameters(self, learning_rate):
        """Update parameters on this device"""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

# Neural Network class managing multiple devices
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.num_devices = len(layer_sizes) - 1
        
        print(f"Initializing {self.num_devices} devices")
        self.devices = []
        for i in range(self.num_devices):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            activation = 'relu' if i < self.num_devices - 1 else 'softmax'
            device = Device(input_size, output_size, activation=activation, device_id=i+1)
            self.devices.append(device)

    def forward(self, X, training=True):
        """Forward pass through all devices"""
        A = X
        activations = [A]
        for device in self.devices:
            A = device.forward(A, training)
            activations.append(A)
        return activations

    def backward(self, activations, y_true):
        """Backward pass through all devices"""
        m = y_true.shape[0]
        y_onehot = np.zeros_like(activations[-1])
        y_onehot[np.arange(m), y_true] = 1
        dA = activations[-1] - y_onehot

        for i in reversed(range(len(self.devices))):
            A_prev = activations[i]
            device = self.devices[i]
            dA = device.backward(dA, A_prev)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=256):
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                activations = self.forward(X_batch)
                y_pred = activations[-1]
                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                
                self.backward(activations, y_batch)
                self.update_parameters()
            
            epoch_loss /= n_batches
            
            if epoch % 10 == 0 or epoch == 1:
                train_acc = self.evaluate_accuracy(y_train, self.forward(X_train)[-1])
                val_acc = self.evaluate_accuracy(y_val, self.forward(X_val)[-1])
                print(f"Epoch {epoch:3d}: Loss = {epoch_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    def compute_loss(self, y_true, y_pred):
        return cross_entropy_loss(y_true, y_pred)

    def update_parameters(self):
        """
        Update parameters of all devices.
        """
        for i, device in enumerate(self.devices):
            device.update_parameters(self.learning_rate)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        activations = self.forward(X)
        y_pred = activations[-1]
        return np.argmax(y_pred, axis=1)

    def evaluate_accuracy(self, y_true, y_pred):
        """
        Evaluate accuracy given true labels and predicted probabilities.
        """
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y_true)
        return accuracy

# Utility function to load and preprocess MNIST data
def load_mnist():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'].astype(np.float32) / 255.0  # Normalize to [0, 1]
    y = mnist['target'].astype(int)
    print(f"Loaded {len(X)} samples")
    return X, y

# Main function to run the training
def main():
    print("Loading MNIST dataset...")
    X, y = load_mnist()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    layer_sizes = [784, 128, 64, 10]
    print(f"Creating neural network with {len(layer_sizes)-1} devices")
    nn = NeuralNetwork(layer_sizes, learning_rate=0.1)
    
    nn.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=256)

if __name__ == "__main__":
    main()
