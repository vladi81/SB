import torch.nn as nn
import torch.nn.functional as F


class VeryGoodCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        buf = F.relu(self.fc1(input))
        buf = self.conv1(buf)
        buf = self.pool(F.relu(buf))
        buf = F.relu(self.fc2(buf))
        buf = self.conv2(buf)
        buf = self.pool(F.relu(buf))
        buf = torch.flatten(buf, 1)
        out = F.relu(self.fc3(buf))
        return out
