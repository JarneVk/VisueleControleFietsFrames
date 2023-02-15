import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #input 40x40
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16, kernel_size=(3,3),stride=2,padding=1), # size out 20x20
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32, kernel_size=(3,3),stride=2,padding=1), # size out 10x10
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=(3,3),stride=2,padding=1), # size out 5x5
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=(5,5)), # size out 1x1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(5,5)), #out 5x5
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=2,padding=1,output_padding=1), #10x10
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=(3,3),stride=2,padding=1,output_padding=1), #20x20
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=(3,3),stride=2,padding=1,output_padding=1),  #20x20
            nn.Sigmoid()
        )


    def forward(self,x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec