import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Set the hyperparameters for the data
NUM_CLASSES = 3
NUM_FEATURES = 13
RANDOM_SEED = 27

# Import the data
data = load_wine()

X = torch.from_numpy(data.data).type(torch.float)
y = torch.from_numpy(data.target).type(torch.float)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size=0.2,
                                                    random_state=RANDOM_SEED)

