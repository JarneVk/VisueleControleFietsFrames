import numpy as np
import torch
import torch.nn as nn
import torch.utils.data 
from torchvision import datasets, models, transforms
import torch.optim as optim
import model_auto as AutoEncModel
from torch.autograd import Variable
from matplotlib import pyplot as plt

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
import os
import PIL, numpy,cv2
import copy

import clustering
from torchsummary import summary

IMSIZE = 80
MODEL = 3
#voor model 3
CHANNELBASE = 48
LATENTDIM = 384 #64,128,256,384

conf_file = open("python/CV/autoEncoders/NetorkConf.txt","w")
conf_file.write(f"{MODEL};{CHANNELBASE};{LATENTDIM}")
conf_file.close()

EARLYSTOP = True

KLUSERING = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Plotting ():
    def __init__(self) -> None:
        self.testingLoss = None
        self.trainingLoss = None
        self.lossGood = None
        self.lossBad = None
        self.lossSmal = None
        self.channelbase = None
        self.latendim = None
        self.model_num = None
    
    def setTrainResults(self,trainingLoss:list,testingLoss:list,model_num:int,channelbas=None,latendim=None):
        self.trainingLoss = trainingLoss
        self.testingLoss = testingLoss
        self.channelbase = channelbas
        self.latendim = latendim
        self.model_num = model_num
    
    def setValidatResults(self,goodloss:list,badloss:list,smalloss:list = None):
        self.lossGood = goodloss
        self.lossBad = badloss
        self.lossSmal = smalloss

    def MakeResults(self):
        x=0
        for _ in os.listdir('python/CV/autoEncoders/trainingLog/autoTraining3'):
            x+=1
        if self.trainingLoss != None and self.testingLoss != None:
            ax = plt.gca()
            ax.set_ylim([0, 0.05])
            plt.plot(self.trainingLoss,label="training")
            plt.plot(self.testingLoss,label="testing")
            plt.xlabel('itterations')
            plt.ylabel('loss')
            plt.title('loss_progress')
            plt.legend(loc='upper right')
            plt.savefig('python/CV/autoEncoders/trainingLog/autoTraining3/lossLog_'+str(x)+'.png')
            # plt.show()
            plt.clf()

        if self.lossBad != None and self.lossGood != None:
            plt.hist(self.lossGood, bins=50,label='normal',alpha = 0.5)
            plt.hist(self.lossBad, bins=50, label='anomaly',alpha = 0.5)
            if self.lossSmal != None:
                plt.hist(self.lossSmal,bins=50, label='small anomaly',alpha = 0.5)
            plt.legend(loc='upper right')
            plt.title(f"autoencoder {self.model_num} loss : c_base {self.channelbase} | latent {self.latendim} ") #: c_base {self.channelbase} | latent {self.latendim}
            plt.savefig('python/CV/autoEncoders/trainingLog/autoTraining3/lossLog_'+str(x)+'hist.png')
            # plt.show()s
            plt.clf()

plotting = Plotting()

#################################################################################################################
#
#                                       load dataset
#
#################################################################################################################

def LoadDataset(dir_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), #convert to grayscale image
        transforms.Resize((IMSIZE,IMSIZE)),
        transforms.ToTensor()
        ])
    batch_size = 32
    dataset = datasets.ImageFolder(dir_path, transform=transform)
    print("dataset has a size of : "+str(len(dataset)))
    train_aant = int(len(dataset)*0.8)
    test_aant = len(dataset)-train_aant

    train_x, test_y = torch.utils.data.random_split(dataset,[train_aant,test_aant])

    train_dataloader = torch.utils.data.DataLoader(train_x, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_y, batch_size=batch_size, shuffle=True)
    print(f"data has been splitted ---- train :{len(train_x)} , test :{len(test_y)}")
    return train_dataloader, test_dataloader,len(train_x),len(test_y)

#################################################################################################################
#
#                                       Training
#
#################################################################################################################


def trainAutoEncoder(dataset_train,t_size,dataset_test,test_size,epochs,model_num=MODEL,base_channel_size=CHANNELBASE,latent_dim=LATENTDIM):
    print("[INFO] initializing the model...")
    model = None
    if model_num == 1:
        model = AutoEncModel.Autoencoder(num_input_chennels=1).to(device)
    elif model_num == 2:
       model = AutoEncModel.Autoencoder_model2(num_input_chennels=1).to(device)
    elif model_num == 3:
        model = AutoEncModel.Autoencoder_model3(base_channel_size,latent_dim,num_input_channels=1).to(device)
    elif model_num == 4:
        model = AutoEncModel.Autoencoder_model4(base_channel_size,latent_dim,num_input_channels=1).to(device) 
    else:
       raise IndexError ('fout argument model_num')
    # summary(model, input_size=(1, IMSIZE, IMSIZE))
    print(model)
    es = EarlyStopping()

    opt = optim.Adam(model.parameters(), lr=0.0005)

    lossFn = nn.MSELoss()
    # lossFn = nn.L1Loss()
    # lossFn = nn.SmoothL1Loss()

    print("[INFO] start training the model...")
    loss_training = []
    loss_testing = []
    for e in range(0, epochs):
        model.train()
        totalTrainLoss = 0
        # loop over the training set
        for (x, _) in dataset_train:
            x = x.to(device)
            decoded,_ = model(x)
            loss = lossFn(decoded, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss.item() * x.size(0)
        loss_training.append(totalTrainLoss/t_size)
        print("[training] Loss of epoch {epoch} : {loss:.6f}".format(epoch=e+1,loss=totalTrainLoss/t_size))

        model.eval()
        totalTestLoss = 0
        for (x, _) in dataset_test:
            x = x.to(device)
            decoded,_ = model(x)
            loss = lossFn(decoded, x)

            opt.zero_grad()
            loss.backward()

            totalTestLoss += loss.item() * x.size(0)
        loss_testing.append(totalTestLoss/test_size)
        print("[testing_] Loss of epoch {epoch} : {loss:.6f} | earlystop : {es}".format(epoch=e+1,loss=totalTestLoss/test_size,es=es.status))
        if es(model,totalTestLoss/test_size) and EARLYSTOP == True:
            plotting.setTrainResults(loss_training,loss_testing,model_num=model_num,channelbas=base_channel_size,latendim=latent_dim)
            return es.best_model
        

    plotting.setTrainResults(loss_training,loss_testing,model_num=model_num,channelbas=base_channel_size,latendim=latent_dim)

    return es.best_model

#################################################################################################################
#
#                                       validate dataset
#
#################################################################################################################

def validate(model,dir_good,dir_fouten=0,smal_dir = None):
    losses_good = []
    losses_bad = []
    losses_small = []

    model.eval()
    lossFn = nn.MSELoss()
    # lossFn = nn.L1Loss()
    # lossFn = nn.SmoothL1Loss()
    transform = transforms.Compose([
          transforms.Grayscale(num_output_channels=1),
          transforms.Resize((IMSIZE,IMSIZE)),
          transforms.ToTensor()
          ])

    #loss van echte fouten
    if dir_fouten ==  0:
        print('geen fouten dir meegegeven')
    else:
        
        dataset = datasets.ImageFolder(dir_fouten, transform=transform)
        it = 0
        totloss = 0
        list_fouten = []
        for i in range(len(dataset)):
            img_org,_ = dataset[i]
            img = Variable(img_org, requires_grad=True)
            img = img.unsqueeze(0)
            image = img.to(device)
            with torch.no_grad():
                deco,latent = model(image)
                if KLUSERING:
                    list_fouten.append(latent.cpu().detach().numpy())
                loss = lossFn(deco, image)
                losses_bad.append(loss.item())
                totloss += loss
            it+=1
        print('[testing] average loss faults : {loss:.6f}'.format(loss=totloss/it))

    if smal_dir ==  None:
        print('geen smal dir meegegeven')
    else:
        
        dataset = datasets.ImageFolder(smal_dir, transform=transform)
        it = 0
        totloss = 0
        list_small_fouten = []
        for i in range(len(dataset)):
            img_org,_ = dataset[i]
            img = Variable(img_org, requires_grad=True)
            img = img.unsqueeze(0)
            image = img.to(device)
            with torch.no_grad():
                deco,latent = model(image)
                if KLUSERING:
                    list_small_fouten.append(latent.cpu().detach().numpy())
                loss = lossFn(deco, image)
                losses_small.append(loss.item())
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
    list_good = []
    for i in range(len(dataset)):
        img,_ = dataset[i]
        img = Variable(img, requires_grad=True)
        img = img.unsqueeze(0)
        image = img.to(device)
        with torch.no_grad():
            deco,latent = model(image)
            if KLUSERING:
                list_good.append(latent.cpu().detach().numpy())
            loss = lossFn(deco, image)
            losses_good.append(loss.item())
            totloss += loss
        it+=1

    print('[testing] average loss good : {loss:.6f}'.format(loss=totloss/it))

    plotting.setValidatResults(losses_good,losses_bad,losses_small)
    plotting.MakeResults()

    if KLUSERING:
        clustering.Kluster(list_good,list_fouten)

    return losses_good,losses_bad,losses_small

def calcRecal_and_precission(fp:int,fn:int,tp:int):
    try:
        precission = tp/(tp+fp)
        recal = tp/(tp+fn)
    except ZeroDivisionError:
        precission = 0
        recal = 0
    return precission,recal

def getPrecisionRecal(losses_good,losses_bad,losses_small):

    precisions = []
    recalls = []
    tp=0
    fp=0
    fn=0
    tresh = 0
    while tresh<0.01:
        for g in losses_good:
            if g>tresh:
                fp +=1

        for b in losses_bad:
            if b>tresh:
                tp+=1
            else:
                fn+=1
        
        precission,recal = calcRecal_and_precission(fp,fn,tp)
        precisions.append(precission)
        recalls.append(recal)
        tresh+=0.0005

    best = 0
    best_idx = 0
    for i in range(len(precisions)):
        if precisions[i]*recalls[i]>best:
            best = precisions[i]*recalls[i]
            best_idx = i
    
    print(f"best presicion: {precisions[best_idx]} | recal: {recalls[best_idx]}")
    print(f"at threshhold {0.005*best_idx}")


    ax = plt.gca()
    # ax.set_ylim([0.8, 1])
    # ax.set_xlim([0.5, 1])  
    plt.plot(recalls,precisions,label="precision-recall")
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PR-curve')
    plt.show()


class EarlyStopping():
  def __init__(self, patience=20, min_delta=0, restore_best_weights=True):
    self.patience = patience
    self.min_delta = min_delta
    self.restore_best_weights = restore_best_weights
    self.best_model = None
    self.best_loss = None
    self.counter = 0
    self.status = ""
    
  def __call__(self, model, val_loss):
    if self.best_loss == None:
      self.best_loss = val_loss
      self.best_model = copy.deepcopy(model)
    elif self.best_loss - val_loss > self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
      self.best_model.load_state_dict(model.state_dict())
    elif self.best_loss - val_loss < self.min_delta:
      self.counter += 1
      if self.counter >= self.patience:
        self.status = f"Stopped on {self.counter}"
        if self.restore_best_weights:
          model.load_state_dict(self.best_model.state_dict())
        return True
    self.status = f"{self.counter}/{self.patience}"
    return False
  
  def getBestModel(self):
      return self.best_model


if __name__ == '__main__':
        DIR_PATH = "dataset_autoenc/good_dir"
        train_dataloader, test_dataloader,tsize,valsize = LoadDataset(DIR_PATH)

        # ittr = iter(train_dataloader)
        # immg,label = ittr._next_data()
        # print(torch.min(immg),torch.max(immg))
        epochs = input("geef een aantal epochs: ")
        model = trainAutoEncoder(train_dataloader,tsize,test_dataloader,valsize,int(epochs))
        good,bad,small = validate(model,"dataset_autoenc/val_dir_train",'dataset_autoenc/bad_dir','dataset_autoenc/smal_dir')
        torch.save(model.state_dict(), 'python/CV/autoEncoders/best_weights.h5')
        
        getPrecisionRecal(good,bad,small)



