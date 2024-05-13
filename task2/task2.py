import torch
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
        # input: nx3x32x32
        buf = self.conv1(input) # -> nx6x28x28
        buf = self.pool(F.relu(buf)) # -> nx6x14x14
        buf = self.conv2(buf) # -> nx16x10x10
        buf = self.pool(F.relu(buf)) # -> nx16x5x5
        buf = torch.flatten(buf, 1) # -> nx400
        buf = F.relu(self.fc1(buf))
        buf = F.relu(self.fc2(buf))
        out = F.relu(self.fc3(buf))
        # out: nx10
        return out


cnn = VeryGoodCNN()
img = torch.randn(1, 3, 32, 32)
out = cnn(img)
