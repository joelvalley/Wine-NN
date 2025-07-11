import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pathlib import Path

# Set the hyperparameters for the data
NUM_CLASSES = 3
NUM_FEATURES = 13
RANDOM_SEED = 27
LEARNING_RATE = 0.001

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
    def __init__(self, input_features: int, output_features: int, hidden_units=32):
        """Initializes multi-class classification model.
        
        Args:
            input_features (int): Num of input features to the model
            output_features (int): Num of output features (number of output classes)
            hidden_units (int): Num of hidden units between layers, default 32
        """
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_features, out_features=hidden_units)   # Input layer
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)   # Hidden layer
        self.layer_3 = nn.Linear(in_features=hidden_units, out_features=hidden_units)   # Hidden layer
        self.layer_4 = nn.Linear(in_features=hidden_units, out_features=output_features)    # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):   # x -> layer_1 -> layer_2 -> layer_3 -> output
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x
    
# Create model
model = WineModel(input_features=13,
                  output_features=3, 
                  hidden_units=64).to(device)

# Setup loss fn & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)

# Calculate prediction accuracy fn
def accuracy_fn(y_true, y_pred):
    """Computes the accuracy of the predictions to the actual label"""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc

# Set seeds
torch.manual_seed(27)
torch.mps.manual_seed(27)
np.random.seed(27)

# Set the number of epochs
epochs = 1501

# Put the data on the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Training & eval loop
for epoch in range(epochs):
    model.train()

    # 1. Forward pass
    y_logits = model(X_train)
    y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)    # Logits -> pred probs -> preds

    #2. Calculate loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)

    #3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward (backpropagation)
    loss.backward()

    # 5. Optimizer step (gradient descent)
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model(X_test)
        test_pred = torch.argmax(torch.softmax(test_logits, dim=1), dim=1)

        # 2. Calculate loss
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, 
                               y_pred=test_pred)

        # 3. Print out what's happenin'
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")

# Make predictions
model.eval()
with torch.inference_mode():
    # 1. Forward pass
    logits = model(X_test)
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

# Save the model

# 1. Create model's directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "wine_model_0.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model's state dict
print(f"Saving the model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), 
           f=MODEL_SAVE_PATH)

# Function to convert the data using PCA and plot
def pca_and_plot_fn(model: nn.Module, 
                    X: torch.Tensor,
                    y_true: torch.Tensor, 
                    y_preds: torch.Tensor, 
                    title_1: str, 
                    title_2: str):
    """Performs principled component analysis on the datasets and plots the results"""

    # Put data on the cpu
    model.to("cpu")
    X, y_true, y_preds = X.to("cpu"), y_true.to("cpu"), y_preds.to("cpu")

    # Transform the data with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    scatter_1 = ax[0].scatter(x=X_pca[:, 0], y=X_pca[:, 1], c=y_true, cmap="RdYlBu", edgecolors="k")
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[0].set_title(title_1)

    handles_1, _ = scatter_1.legend_elements()
    ax[0].legend(handles_1, ["Class 1", "Class 2", "Class 3"], loc="upper right")

    scatter_2 = ax[1].scatter(x=X_pca[:, 0], y=X_pca[:, 1], c=y_preds, cmap="RdYlBu", edgecolors="k")
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC2")
    ax[1].set_title(title_2)

    handles_2, _ = scatter_2.legend_elements()
    ax[1].legend(handles_2, ["Class 1", "Class 2", "Class 3"], loc="upper right")

    plt.show()

# Plot the data
pca_and_plot_fn(model, X_test, y_test, pred, "PCA of Wine Test Dataset", "PCA of Model's Test Predictions")