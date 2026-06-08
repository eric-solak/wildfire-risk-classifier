import torch
from torch import nn


class RiskClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.bn3(self.fc3(x))
        x = self.relu(x)
        x = self.dropout2(x)
        return self.fc4(x)
