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

class DeviceVisualizer:
    def __init__(self, num_devices):
        self.num_devices = num_devices
        self.current_state = ['idle'] * num_devices
        self.metrics = {'loss': 0, 'accuracy': 0}
        self.clear_line = '\r' + ' ' * 150 + '\r'
    
    def _draw_device(self, device_id, state):
        if state == 'forward':
            return f"\033[94m[D{device_id}→]\033[0m"
        elif state == 'backward':
            return f"\033[91m[D{device_id}←]\033[0m"
        else:
            return f"[D{device_id} ]"
    
    def update(self, loss=None, accuracy=None, state=None, device_id=None):
        if loss is not None:
            self.metrics['loss'] = loss
        if accuracy is not None:
            self.metrics['accuracy'] = accuracy
        if state is not None and device_id is not None:
            self.current_state[device_id-1] = state
            
        # Create device pipeline visualization
        devices = ' → '.join(self._draw_device(i+1, state) 
                           for i, state in enumerate(self.current_state))
        
        # Add metrics if available
        metrics = f" | Loss: {self.metrics['loss']:.4f} | Acc: {self.metrics['accuracy']:.4f}"
        
        # Full pipeline visualization
        pipeline = f"{self.clear_line}{devices}{metrics}"
        sys.stdout.write(pipeline)
        sys.stdout.flush()

class Device:
    def __init__(self, input_size, output_size, activation='relu', device_id=0, visualizer=None):
        self.device_id = device_id
        self.visualizer = visualizer
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation
        self.dropout_rate = 0.2
        self.mask = None
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
        self.beta = 0.9

    def forward(self, A_prev, training=True):
        """Forward pass with dropout"""
        if self.visualizer:
            self.visualizer.update(state='forward', device_id=self.device_id)
            time.sleep(0.1)  # Add small delay to make visualization visible
            
        self.Z = np.dot(A_prev, self.W) + self.b
        if self.activation == 'relu':
            self.A = relu(self.Z)
            if training:
                self.mask = np.random.rand(*self.A.shape) > self.dropout_rate
                self.A *= self.mask
                self.A /= (1 - self.dropout_rate)
        elif self.activation == 'softmax':
            self.A = softmax(self.Z)
            
        if self.visualizer:
            self.visualizer.update(state='idle', device_id=self.device_id)
        return self.A

    def backward(self, dA, A_prev):
        """Simulate backward pass on this device"""
        if self.visualizer:
            self.visualizer.update(state='backward', device_id=self.device_id)
            time.sleep(0.1)
            
        m = A_prev.shape[0]
        if self.activation == 'relu':
            dZ = dA * relu_derivative(self.Z)
        elif self.activation == 'softmax':
            dZ = dA
        
        self.dW = np.dot(A_prev.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        self.dA_prev = np.dot(dZ, self.W.T)
        
        if self.visualizer:
            self.visualizer.update(state='idle', device_id=self.device_id)
        return self.dA_prev

    def update_parameters(self, learning_rate):
        """Update parameters with momentum"""
        self.vW = self.beta * self.vW + (1 - self.beta) * self.dW
        self.vb = self.beta * self.vb + (1 - self.beta) * self.db
        
        self.W -= learning_rate * self.vW
        self.b -= learning_rate * self.vb

# Neural Network class managing multiple devices
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.num_devices = len(layer_sizes) - 1  # Number of layers = number of devices
        self.visualizer = DeviceVisualizer(self.num_devices)
        
        print(f"\n\033[92mInitializing {self.num_devices} Devices:\033[0m")
        
        # Create devices (including input layer)
        self.devices = []
        for i in range(self.num_devices):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            activation = 'relu' if i < self.num_devices - 1 else 'softmax'
            
            device = Device(input_size, output_size, 
                          activation=activation, 
                          device_id=i+1,
                          visualizer=self.visualizer)
            self.devices.append(device)
            print(f"Device {i+1}: {input_size} → {output_size} ({activation})")
        print()

    def forward(self, X, training=True):
        """Forward pass through all devices"""
        A = X
        activations = [A]
        for device in self.devices:
            A = device.forward(A, training)
            activations.append(A)
        return activations

    def compute_loss(self, y_true, y_pred):
        return cross_entropy_loss(y_true, y_pred)

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

    def update_parameters(self):
        """
        Update parameters of all devices.
        """
        for i, device in enumerate(self.devices):
            device.update_parameters(self.learning_rate)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, print_every=10, batch_size=256):
        print(f"\033[92mStarting distributed training across {self.num_devices} devices\033[0m")
        print(f"[D1] → [D2] → [D3] → [D4] (Blue: Forward Pass, Red: Backward Pass)")
        
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in tqdm(range(1, epochs + 1), desc="Training"):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass through devices
                activations = self.forward(X_batch, training=True)
                y_pred = activations[-1]
                batch_loss = self.compute_loss(y_batch, y_pred)
                batch_acc = self.evaluate_accuracy(y_batch, y_pred)
                
                # Update visualization with current metrics
                self.visualizer.update(loss=batch_loss, accuracy=batch_acc)
                
                # Backward pass and update
                self.backward(activations, y_batch)
                self.update_parameters()

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
    # Load data
    print("Loading MNIST dataset...")
    X, y = load_mnist()
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define network architecture with explicit input layer
    layer_sizes = [784, 128, 64, 10]  # Input → Hidden → Hidden → Output
    print(f"\033[92mCreating distributed neural network across {len(layer_sizes)} devices\033[0m")
    nn = NeuralNetwork(layer_sizes, learning_rate=0.1)
    
    nn.train(X_train, y_train, X_val, y_val, 
            epochs=100, 
            print_every=10, 
            batch_size=256)
    
    # Final evaluation
    y_val_pred = nn.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"\nFinal Validation Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()
