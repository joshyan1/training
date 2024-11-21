import threading
import torch
import logging
import argparse
from models.model import ResNet, ResidualBlock

optimizer_function = lambda new_model: torch.optim.SGD(new_model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)
device = torch.device("mps")

class Leader():
    def __init__(self, learners, model, epochs, batch_size):
        self.lock = threading.Lock()
        self.max_learners = learners
        self.learners = []
        self.gradients = []
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer_function(self.model)
        logging.info("Leader service initialized")
        self.initialize_leader()

    def initialize_leader(self):
        pass
        #set up GRPC

        #start server

    def AddLearner(self):
        #set up Learner
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Training Leader')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--learners', type=int, default=0, metavar='L',
                        help='Learner Count')
    args = parser.parse_args()

    model = ResNet(ResidualBlock, layers=[2,2,2,2]).to(device)
    leader = Leader(args.learners, model, args.epochs, args.batch_size)


