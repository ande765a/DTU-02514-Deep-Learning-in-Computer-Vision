import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, n_features, k_size=3):
        super(ResNetBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(n_features, n_features, k_size, padding=k_size//2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=n_features),
            nn.Conv2d(n_features, n_features, k_size, padding=k_size//2)
        )

    def forward(self, x):
        out = F.relu(self.res_block(x) + x)
        return out


class NumDetector(nn.Module):
    def __init__(self):
        super(NumDetector, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),

            ResNetBlock(n_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),

            ResNetBlock(n_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            ResNetBlock(n_features=128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),

            nn.Conv2d(in_channels=256, out_channels=11, kernel_size=1),
            nn.Softmax2d()
        )
    
    def forward(self, x):
        x = self.cnn(x)
        return x
