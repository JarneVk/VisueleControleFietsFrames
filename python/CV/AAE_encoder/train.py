# inspiration : https://github.com/bfarzin/pytorch_aae/blob/master/main_aae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import datasets

import model as AAE

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x) 

def trainAAE(data_loader):
    EPS = 1e-15
    z_red_dims = 120
    Q = AAE.Q_net(1600,1000,z_red_dims).cuda()
    P = AAE.P_net(1600,1000,z_red_dims).cuda()
    D_gauss = AAE.D_net_gauss(500,z_red_dims).cuda()

    # Set learning rates
    gen_lr = 0.0001
    reg_lr = 0.00005

    #encode/decode optimizers
    optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
    optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
    #regularizing optimizers
    optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
    optim_D = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)

    data_iter = iter(data_loader)
    iter_per_epoch = len(data_loader)
    total_step = 2000

    # Start training
    for step in range(total_step):

        # Reset the data_iter
        if (step+1) % iter_per_epoch == 0:
            data_iter = iter(data_loader)

        # Fetch the images and labels and convert them to variables
        images, labels = next(data_iter)
        images, labels = to_var(images.view(images.size(0), -1)), to_var(labels)

        #reconstruction loss
        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        z_sample = Q(images)   #encode to z
        X_sample = P(z_sample) #decode to X reconstruction
        recon_loss = F.binary_cross_entropy(X_sample+EPS,images+EPS)

        recon_loss.backward()
        optim_P.step()
        optim_Q_enc.step()

        # Discriminator
        ## true prior is random normal (randn)
        ## this is constraining the Z-projection to be normal!
        Q.eval()
        z_real_gauss = Variable(torch.randn(images.size()[0], z_red_dims) * 5.).cuda()
        D_real_gauss = D_gauss(z_real_gauss)

        z_fake_gauss = Q(images)
        D_fake_gauss = D_gauss(z_fake_gauss)

        D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))

        D_loss.backward()
        optim_D.step()

        # Generator
        Q.train()
        z_fake_gauss = Q(images)
        D_fake_gauss = D_gauss(z_fake_gauss)
        
        G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))

        G_loss.backward()
        optim_Q_gen.step()   

        if (step+1) % 100 == 0: #every 100 steps
            print("recon_loss : {}    | descriminator_loss : {}     | generator_loss : {}".format(recon_loss.data.item(),D_loss.data.item(),G_loss.data.item()))

    torch.save(Q.state_dict(),'Q_encoder_weights.pt')

def eval_AAE(dataset_test_good):
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test_good, 
                                          batch_size=10000, 
                                               shuffle=True)
    data_iter_test = iter(data_loader_test)    
    # Fetch the images and labels and convert them to variables
    images, labels = next(data_iter_test)
    images, labels = to_var(images.view(images.size(0), -1)), to_var(labels)

    outputs = net(Q(images))
    # outputs = net(images)

    # Compute accuracy
    _, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()

    print(accuracy.data[0])

if __name__ == '__main__':
    dir_path = 'dataset_autoenc/good_dir'
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), #convert to grayscale image
        transforms.Resize((40,40)),
        transforms.ToTensor()
        ])
    dataset = datasets.ImageFolder(dir_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=100,shuffle=True)

    trainAAE(data_loader)