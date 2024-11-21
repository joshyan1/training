import os
import requests
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import logging

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

MODEL_SAVE_PATH = "simple_model.pth"

def load_model():
    # Initialize the model architecture
    model = SimpleModel()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()  # Set the model to evaluation mode
    return model

if __name__ == "__main__":
    # Load the model
    model = load_model()
    print("Model successfully loaded!")

    # Example usage: Inference on dummy data
    dummy_input = torch.rand(1, 28, 28)  # Random input tensor simulating a single MNIST image
    output = model(dummy_input)
    print("Model output:", output)
