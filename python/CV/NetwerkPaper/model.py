from torch import nn, optim
import torch


class PaperNet(nn.Module):
    def __init__(self,numChannels, classes):
        super(PaperNet, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.convolutional = nn.Sequential(
            # 8@5x5 conv layer
            nn.Conv2d(in_channels=numChannels,out_channels=8,kernel_size=(5,5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            # 8@5x5 conv layer
            nn.Conv2d(in_channels=8,out_channels=8,kernel_size=(5,5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
            # 8@3x3 conv layer
            nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 8@3x3 conv layer
            nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 8@3x3 conv layer
            nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3)
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=8*3*3,out_features=512),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=256,out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64,out_features=classes),
            nn.LogSoftmax()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.convolutional(x)
        flat = torch.flatten(conv,1)
        linear = self.linear(flat)
        return linear