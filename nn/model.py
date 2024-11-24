import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import sys
import time

print("Imported necessary libraries")

class Device:
    def __init__(self, input_size, output_size, activation='relu', device_id=0):
        self.device_id = device_id
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize weights using PyTorch
        self.W = torch.randn(input_size, output_size, device=self.device) * torch.sqrt(torch.tensor(2.0 / input_size))
        self.b = torch.zeros(1, output_size, device=self.device)
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)
        
        self.activation = activation

    def forward(self, A_prev, training=True):
        """Forward pass on this device"""
        self.A_prev = A_prev
        self.Z = torch.mm(A_prev, self.W) + self.b
        
        if self.activation == 'relu':
            self.A = F.relu(self.Z)
        elif self.activation == 'softmax':
            self.A = F.softmax(self.Z, dim=1)
        return self.A

    def backward(self, dA, A_prev):
        """Backward pass on this device"""
        m = A_prev.size(0)
        
        if self.activation == 'relu':
            dZ = dA * (self.Z > 0)
        elif self.activation == 'softmax':
            dZ = dA
        
        self.dW = torch.mm(A_prev.t(), dZ) / m
        self.db = torch.sum(dZ, dim=0, keepdim=True) / m
        self.dA_prev = torch.mm(dZ, self.W.t())
        return self.dA_prev

    def update_parameters(self, learning_rate):
        """Update parameters on this device"""
        with torch.no_grad():
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.num_devices = len(layer_sizes) - 1
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
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
        m = y_true.size(0)
        y_onehot = torch.zeros_like(activations[-1], device=self.device)
        y_onehot.scatter_(1, y_true.unsqueeze(1), 1)
        dA = activations[-1] - y_onehot

        for i in reversed(range(len(self.devices))):
            A_prev = activations[i]
            device = self.devices[i]
            dA = device.backward(dA, A_prev)

    def train(self, train_loader, val_loader, epochs=100, batch_size=256):
        print("\nStarting training...")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print("-" * 100)
        
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.evaluate(val_loader)
            
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            print("-" * 100)
            
            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    def train_epoch(self, train_loader, epoch):
        total_loss = 0
        total_acc = 0
        n_batches = len(train_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            activations = self.forward(data)
            y_pred = activations[-1]
            
            # Compute loss
            loss = F.cross_entropy(y_pred, target)
            acc = (y_pred.argmax(dim=1) == target).float().mean()
            
            # Backward pass
            self.backward(activations, target)
            
            # Update parameters
            for device in self.devices:
                device.update_parameters(self.learning_rate)
            
            total_loss += loss.item()
            total_acc += acc.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"\rEpoch {epoch} [{batch_idx+1}/{n_batches}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {acc.item():.4f}", end="")
        
        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches
        print(f"\nEpoch {epoch} - Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}")

    def evaluate(self, val_loader):
        total_loss = 0
        total_acc = 0
        n_batches = len(val_loader)
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                activations = self.forward(data, training=False)
                y_pred = activations[-1]
                
                loss = F.cross_entropy(y_pred, target)
                acc = (y_pred.argmax(dim=1) == target).float().mean()
                
                total_loss += loss.item()
                total_acc += acc.item()
        
        return total_loss / n_batches, total_acc / n_batches

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000)
    
    layer_sizes = [784, 128, 64, 10]
    print(f"Creating neural network with architecture: {layer_sizes}")
    nn = NeuralNetwork(layer_sizes, learning_rate=0.1)
    
    nn.train(train_loader, val_loader, epochs=100)

if __name__ == "__main__":
    main()
