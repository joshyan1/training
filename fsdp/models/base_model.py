class Model(ABC, nn.Module):
    """Base model class for all models."""

    def __init__(self):
        super(Model, self).__init__()

    @abstractmethod
    def forward(self, x):
        """Forward pass."""
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs):
        """Custom training logic."""
        pass