import torch
import torch.nn as nn

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim

###################################################################################################################################################
#
#                                                   Model 1 
#
###################################################################################################################################################

class Autoencoder(nn.Module):
    def __init__(self,num_input_chennels) -> None:
        super().__init__()
        #input 40x40
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=num_input_chennels,out_channels=16, kernel_size=(3,3),stride=2,padding=1), # size out 20x20
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
            nn.ConvTranspose2d(in_channels=16,out_channels=num_input_chennels,kernel_size=(3,3),stride=2,padding=1,output_padding=1),  #20x20
            nn.Sigmoid()
        )


    def forward(self,x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
    
###################################################################################################################################################
#
#                                                   Model 2 
#
###################################################################################################################################################


class Autoencoder_model2(nn.Module):
    def __init__(self,num_input_chennels) -> None:
        super().__init__()
        #input 80x80
        self.encoder1 = nn.Sequential(
            nn.Conv2d(num_input_chennels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,return_indices=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,return_indices=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=2,stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,return_indices=True)
        )

        self.decoderUnpool = nn.MaxUnpool2d(kernel_size=3,stride=2)        

        self.decoder1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,padding=2,stride=2,output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256,out_channels=384,kernel_size=3,padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=384,out_channels=192,kernel_size=3,padding=1)
        )
        self.decoder2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=192,out_channels=64,kernel_size=5,padding=2)
        ) 
        self.decoder3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=num_input_chennels,kernel_size=11,stride=4,padding=2,output_padding=1),
            nn.Sigmoid()
        )


    def forward(self,x: torch.Tensor):
        x,ind_1 = self.encoder1(x)
        x,ind_2 = self.encoder2(x)
        enc,ind_3 = self.encoder3(x)

        x = self.decoderUnpool(enc,ind_3)
        x = self.decoder1(x)
        x = self.decoderUnpool(x,ind_2)
        x = self.decoder2(x)
        x = self.decoderUnpool(x,ind_1)
        dec = self.decoder3(x)

        return dec
    
###################################################################################################################################################
#
#                                                       Model 3 
#       credits: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
###################################################################################################################################################
class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=11, padding=2, stride=4),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=5, padding=2, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=2, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 36 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 36 * c_hid), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=2, stride=2
            ),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=5, output_padding=1, padding=2, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=11, output_padding=1, padding=4, stride=4
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 6, 6)
        x = self.net(x)
        return x
    
class Autoencoder_model3(pl.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 80,
        height: int = 80,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)