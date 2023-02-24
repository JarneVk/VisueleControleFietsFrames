import numpy as np
import torch
import torch.nn as nn
import torch.utils.data 
from torchvision import datasets, models, transforms
import torch.optim as optim
import model as AutoEncModel
from torch.autograd import Variable
from matplotlib import pyplot as plt

import PIL, numpy,cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def LoadDataset(dir_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), #convert to grayscale image
        transforms.Resize((40,40)),
        transforms.ToTensor()
        ])
    batch_size = 32
    dataset = datasets.ImageFolder(dir_path, transform=transform)
    print("dataset has a size of : "+str(len(dataset)))
    train_aant = int(len(dataset)*0.9)
    test_aant = len(dataset)-train_aant

    train_x, test_y = torch.utils.data.random_split(dataset,[train_aant,test_aant])

    train_dataloader = torch.utils.data.DataLoader(train_x, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_y, batch_size=batch_size, shuffle=True)
    print(f"data has been splitted ---- train :{len(train_x)} , test :{len(test_y)}")
    return train_dataloader, test_dataloader,len(train_x),len(test_y)

def trainAutoEncoder(dataset_train,t_size,epochs):
    print("[INFO] initializing the model...")
    model = AutoEncModel.Autoencoder().to(device)

    opt = optim.Adam(model.parameters(), lr=0.01)
    lossFn = nn.MSELoss()
    
    for e in range(0, epochs):
        print('Epoch {}/{}'.format(e+1, epochs))
        print('-' * 10)
        model.train()
        totalTrainLoss = 0
        # loop over the training set
        for (x, _) in dataset_train:
            x = x.to(device)
            decoded = model(x)
            loss = lossFn(decoded, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss
        print("[training] Loss of epoch {epoch} : {loss:.6f}".format(epoch=e+1,loss=totalTrainLoss/t_size))
    return model


def validate(model,dir_good,dir_fouten=0):
    losses_good = []
    losses_bad = []

    model.eval()
    lossFn = nn.MSELoss()
    transform = transforms.Compose([
          transforms.Grayscale(num_output_channels=1),
          transforms.Resize((40,40)),
          transforms.ToTensor()
          ])

    #loss van echte fouten
    if dir_fouten ==  0:
        print('geen fouten dir meegegeven')
    else:
        
        dataset = datasets.ImageFolder(dir_fouten, transform=transform)
        it = 0
        totloss = 0
        for i in range(len(dataset)):
            img_org,_ = dataset[i]
            img = Variable(img_org, requires_grad=True)
            img = img.unsqueeze(0)
            image = img.to(device)
            with torch.no_grad():
                deco = model(image)
                loss = lossFn(deco, image)
                losses_bad.append(loss.item())
                totloss += loss
            it+=1
        print('[testing] average loss faults : {loss:.6f}'.format(loss=totloss/it))
    
    #loss van goede delen
    # it = 0
    # totloss = 0
    # for (x, _) in test_dataloader:
    #     x = x.to(device)
    #     decoded = model(x)
    #     loss = lossFn(decoded, x)

    dataset = datasets.ImageFolder(dir_good, transform=transform)
    it = 0
    totloss = 0
    for i in range(len(dataset)):
        img,_ = dataset[i]
        img = Variable(img, requires_grad=True)
        img = img.unsqueeze(0)
        image = img.to(device)
        with torch.no_grad():
            deco = model(image)
            loss = lossFn(deco, image)
            losses_good.append(loss.item())
            totloss += loss
        it+=1

    print('[testing] average loss good : {loss:.6f}'.format(loss=totloss/it))
    return losses_good,losses_bad

def plotResults(loss_good,loss_bad):
    for i in loss_good:
        plt.scatter(i,1.1,color='green')
    for i in loss_bad:
        plt.scatter(i,1,color='red')
    plt.xlim([0, 0.5])
    plt.show()



if __name__ == '__main__':
        DIR_PATH = "dataset_autoenc/good_dir"
        train_dataloader, test_dataloader,tsize,valsize = LoadDataset(DIR_PATH)

        # ittr = iter(train_dataloader)
        # immg,label = ittr._next_data()
        # print(torch.min(immg),torch.max(immg))
        epochs = input("geef een aantal epochs: ")
        model = trainAutoEncoder(train_dataloader,tsize,int(epochs))
        good,bad = validate(model,DIR_PATH,'dataset_autoenc/bad_dir')

        plotResults(good,bad)
