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
    def __init__(self, n_in=3, n_features =16, num_res_blocks=3, block_depth = 1):
        super(ResNet, self).__init__()
        #First conv layers needs to output the desired number of features.
        conv_layers = [
            nn.Conv2d(n_in, n_features, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ]

        
        for i in range(block_depth):
            for _ in range(num_res_blocks):
                conv_layers.append(ResNetBlock(n_features*(i+1)))
            conv_layers.append(nn.Conv2d(n_features*(i+1), n_features*(i+2), kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ReLU())

        conv_layers.append(nn.AvgPool2d(2))
        self.res_blocks = nn.Sequential(*conv_layers)


        self.fc = nn.Sequential(
            nn.Linear(16*16*n_features*(block_depth+1), 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512,1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.res_blocks(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out.view(-1)