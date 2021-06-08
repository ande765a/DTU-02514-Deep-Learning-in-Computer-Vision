import torch
import torch.nn as nn
import numpy as np

class ResNetBlock(nn.Module):
    def __init__(self, n_features):
        super(ResNetBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_features, n_features, 3, padding=1)
        )

    def forward(self, x):
        out = nn.functional.relu(self.res_block(x) + x)
        return out


class ResNet(nn.Module):
    def __init__(self, n_in, n_features, num_res_blocks=3):
        super(ResNet, self).__init__()
        #First conv layers needs to output the desired number of features.
        conv_layers = [
            nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ]

        for _ in range(num_res_blocks):
            conv_layers.append(ResNetBlock(n_features))

        self.res_blocks = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
            nn.Linear(32*32*n_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.res_blocks(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out