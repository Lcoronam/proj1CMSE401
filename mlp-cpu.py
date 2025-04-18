import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Toggle between benchmark and accuracy mode
benchmark_mode = os.getenv("BENCHMARK_MODE") == "1"

if benchmark_mode:
    print("BENCHMARK MODE: Using synthetic 1M-row dataset")
    X, y = make_classification(
        n_samples=1_000_000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    X = StandardScaler().fit_transform(X)
else:
    print("ACCURACY MODE: Using real wine dataset")
    filename = "winequality-white.csv"
    if not os.path.exists(filename):
        print("Dataset not found. Downloading...")
        import urllib.request
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        urllib.request.urlretrieve(url, filename)
    df = pd.read_csv(filename, sep=';')
    df["quality"] = (df["quality"] >= 7).astype(int)
    X = StandardScaler().fit_transform(df.drop("quality", axis=1).values)
    y = df["quality"].values

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# MLP model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Force CPU
device = torch.device("cpu")
print(f"Using device: {device}")
model = MLPClassifier(input_dim=X.shape[1]).to(device)

# Move data to CPU (already is, but for consistency)
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Training setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Full-batch training loop
start = time.time()
for iteration in range(10):
    model.train()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Iteration {iteration} | Loss: {loss.item():.4f}", flush=True)
end = time.time()

print(f"\nTotal training time: {end - start:.2f} seconds on {device}", flush=True)

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = (predictions >= 0.5).float()
    accuracy = (predicted_classes == y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%", flush=True)








