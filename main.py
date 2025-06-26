import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Set the hyperparameters for the data
NUM_CLASSES = 3
NUM_FEATURES = 13
RANDOM_SEED = 27

# Import the dataset from sklearn
data = load_wine()

X = torch.from_numpy(data.data).type(torch.float)
y = torch.from_numpy(data.target).type(torch.float)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size=0.2,
                                                    random_state=RANDOM_SEED)

# Make device agnostic code
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Create the model
class WineModel(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units=50):
        """Initializes multi-class classification model.
        
        Args:
            input_features (int): Num of input features to the model
            output_features (int): Num of output features (number of output classes)
            hidden_units (int): Num of hidden units between layers, default 50
        """
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_features, out_features=hidden_units)   # Input layer
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)   # Hidden layer
        self.layer_3 = nn.Linear(in_features=hidden_units, out_features=output_features)    # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):   # x -> layer_1 -> layer_2 -> layer_3 -> output
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x
    
# Create model
model = WineModel(input_features=13,
                  output_features=3)

# Setup loss fn & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.01)

# Set seeds
torch.manual_seed(27)
torch.mps.manual_seed(27)
np.random.seed(27)