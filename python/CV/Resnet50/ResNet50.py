import numpy as np
import torch
import torch.utils.data 
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import time
from PIL import Image
import joblib
import pandas
import os
import io
import copy
import matplotlib.pyplot as plt


PICTUREHEIGHT = 128
PICTUREWIDTH = 128

CLASSES = 2
LEARNING_RATE = 0.0005
MODEL = 'resnet34'

EARLYSTOP = True

VALDATA = 'ValidataionSet'

class visualise():
    def Visualise(list,title="loss",Save_path='python/CV/Resnet50/trainingLog/default.png'):
        ax = plt.gca()
        ax.set_ylim([0, 1])
        for l in list:
            plt.plot(l)
            plt.xlabel('itterations')
            plt.ylabel('loss')
            plt.title(title)
        plt.savefig(Save_path)
        plt.show()

class EarlyStopping():
  def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
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

class CreateDataset():
    def __init__(self) -> None:
        pass

    def LoadDataset(dir_path):
        transform = transforms.Compose([
          transforms.Resize((PICTUREWIDTH,PICTUREHEIGHT)),
          transforms.ToTensor()
          ])
        batch_size = 16
        dataset = datasets.ImageFolder(dir_path, transform=transform)
        classmapping = dataset.class_to_idx
        print("dataset has a size of : "+str(len(dataset)))
        train_aant = int(len(dataset)*0.80)
        test_aant = len(dataset)-train_aant

        train_x, test_y = torch.utils.data.random_split(dataset,[train_aant,test_aant])

        train_dataloader = DataLoader(train_x, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_y, batch_size=batch_size, shuffle=True)
        print(f"data has been splitted ---- train :{len(train_x)} , test :{len(test_y)}")
        joblib.dump(classmapping,'python/CV/Resnet50/classmapping')

        valdataset = datasets.ImageFolder(VALDATA, transform=transform)
        val_dataloader = DataLoader(valdataset, batch_size=batch_size)

        return train_dataloader, test_dataloader,len(train_x),len(test_y),test_y,val_dataloader,len(valdataset)

class Resnet50_CreateModel():
    def __init__(self,train_dataloader, test_dataloader,val_dataset,num_epochs=3):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.traindata = train_dataloader
        self.testdata = test_dataloader
        self.valdata = val_dataset
        self.num_epochs = num_epochs
        self.classmap = joblib.load('python/CV/Resnet50/classmapping')
        self.learningRate = LEARNING_RATE
        x=0
        for _ in os.listdir('python/CV/Resnet50/trainingLog'):
            x+=1
        self.loging = open('python/CV/Resnet50/trainingLog/log'+str(x)+'.txt','w')
        self.saveLogPict = 'python/CV/Resnet50/trainingLog/log'+str(x)+'.png'

    def setLearningRate(self,newlr):
        self.learningRate = newlr

    def train(self,tsize,valsize,size_val):
        if MODEL == 'resnet34':
            model = models.resnet34(weights='DEFAULT').to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(512, CLASSES).to(self.device)

        elif MODEL == 'resnet101':
            model = models.resnet101(weights='DEFAULT').to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, CLASSES).to(self.device)

        else:
            model = models.resnet50(weights='DEFAULT').to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, CLASSES)).to(self.device)
        

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(),lr=self.learningRate)
        model_trained = self.train_model(model, criterion, optimizer,tsize,valsize,size_val, self.num_epochs)
        return model_trained

    def train_model(self,model, criterion, optimizer,tsize,valsize,size_val,num_epochs):

        t_loss_list = []
        test_loss_list = []
        v_loss_list = []

        earlstop = EarlyStopping()

        print(f"train :{tsize} , test :{valsize}")
        t_start = time.time()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
            start_epoch = time.time()
            for phase in ['train','validation', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                if phase == 'train':
                    for idx, (inputs, labels) in enumerate(self.traindata):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        _, preds = torch.max(outputs, 1)
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / float(tsize) 
                    epoch_acc = running_corrects.double() / float(tsize)
                    t_loss_list.append(epoch_loss)
                    self.loging.write('train_loss:'+str(epoch_loss)+'\n')
                elif phase == 'test':
                    rp=0
                    fp=0
                    fn=0
                    rn = 0
                    for idx, (inputs, labels) in enumerate(self.testdata):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        

                    epoch_loss = running_loss / valsize 
                    epoch_acc = running_corrects.double() / valsize 
                    test_loss_list.append(epoch_loss)
                    self.loging.write('test_loss:'+str(epoch_loss)+'\n')
                    print('ealyStrop: [{}]'.format(earlstop.status))
                    if earlstop(model,epoch_loss) and EARLYSTOP:
                        t_tijd = time.time() - t_start
                        print(f"total time : {t_tijd}")
                        lijst = [t_loss_list,test_loss_list,v_loss_list]
                        title = str(MODEL)+" |  loss: epochs "+str(num_epochs)+" | lr "+str(LEARNING_RATE)+" | earlyStop "+str(EARLYSTOP)
                        visualise.Visualise(lijst,title,self.saveLogPict)
                        return earlstop.getBestModel()
                else :
                    for idx, (inputs, labels) in enumerate(self.valdata):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        

                    epoch_loss = running_loss / size_val
                    epoch_acc = running_corrects.double() / size_val 
                    v_loss_list.append(epoch_loss)
                    self.loging.write('val_loss:'+str(epoch_loss)+'\n')

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))

        t_tijd = time.time() - t_start
        print(f"total time : {t_tijd}")
        lijst = [t_loss_list,test_loss_list,v_loss_list]
        title = str(MODEL)+" |  loss: epochs "+str(num_epochs)+" | lr "+str(LEARNING_RATE)+" | earlyStop "+str(EARLYSTOP)
        visualise.Visualise(lijst,title,self.saveLogPict)
        return earlstop.getBestModel()


class Resnet50_testModel():
    def __init__(self,model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.mapping = joblib.load('python/CV/Resnet50/classmapping')
        self.transform = transforms.Compose([
          transforms.Resize((PICTUREWIDTH,PICTUREHEIGHT)),
          transforms.ToTensor()
          ])

    def loadModel(path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if MODEL == 'resnet34':
            model = models.resnet34(weights=path).to(device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(512, CLASSES).to(device)

        elif MODEL == 'resnet101':
            model = models.resnet101(weights=path).to(device)
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, CLASSES).to(device)

        else:
            model = models.resnet50(weights=path).to(device)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, CLASSES)).to(device)
        model.load_state_dict(torch.load(path))
        return model

    def imageLoader(self,input):
        imsize = PICTUREWIDTH
        loader = transforms.Compose([ transforms.ToTensor(),transforms.Resize(imsize)])
        # image = Image.open(input)
        image = loader(input).float() 

        # transform = transforms.ToTensor()
        # image = transform(input)

        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        return image.cuda()
    
    def predictSingleImage(self,input):
        im_tensor = self.imageLoader(input)
        predict = self.predict(im_tensor)
        print(f"predicted:{predict}")
        return predict

    def predictDataset(self,dir_path):
        rp=0
        fp=0
        fn=0
        rn = 0

        dataset = datasets.ImageFolder(dir_path, transform=self.transform)
        for i in range(len(dataset)):
            img,label = dataset[i]
            img = Variable(img, requires_grad=True)
            img = img.unsqueeze(0)
            image = img.to(self.device)
            prediction = self.predict(image)
            expected = list(self.mapping)[label]
            #print(f"predicted: {prediction} <-> expected: {expected}")
            if prediction == "good":
                if expected == "good":
                    rp+=1
                else:
                    fp+=1
            elif prediction == "bad":
                if expected == "bad":
                    rn+=1
                else:
                    fn+=1
        data = [[rp,fp],[fn,rn]]
        headers=["good", "bad"]
        print("_____|expected")
        print("pred |")
        print(pandas.DataFrame(data, headers, headers))

    def predictDatasetBySetInput(self,dataset):
        rp=0
        fp=0
        fn=0
        rn = 0
        for i in range(len(dataset)):
            img,label = dataset[i]
            img = Variable(img, requires_grad=True)
            img = img.unsqueeze(0)
            image = img.to(self.device)
            prediction = self.predict(image)
            expected = list(self.mapping)[label]
            #print(f"predicted: {prediction} <-> expected: {expected}")
            if prediction == "good":
                if expected == "good":
                    rp+=1
                else:
                    fp+=1
            elif prediction == "bad":
                if expected == "bad":
                    rn+=1
                else:
                    fn+=1
        data = [[rp,fp],[fn,rn]]
        headers=["good", "bad"]
        print("_____|expected")
        print("pred |")
        print(pandas.DataFrame(data, headers, headers))
            


    def predict(self,input):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input)
            pred_idx = pred[0].argmax(0)
            #print(f"classmap:{self.mapping} -> index: {pred_idx.item()}")
            predicted = list(self.mapping)[pred_idx.item()]
        return predicted

if __name__ == '__main__':
    DIR_PATH = "dataset_2"
    input_str = input("test or train? :")
    if input_str == "train":
        #----------------------------------- creating dataset -------------------------------------------
        train_dataloader, test_dataloader,tsize,valsize,test_y,val_dataset,size_val = CreateDataset.LoadDataset(DIR_PATH)
        #-----------------------------------   train model    -------------------------------------------
        epochs = int(input('amount of epochs :'))
        resnet = Resnet50_CreateModel(train_dataloader, test_dataloader,val_dataset,epochs)
        model_trained = resnet.train(tsize,valsize,size_val)
        #-----------------------------------    save model    -------------------------------------------
        torch.save(model_trained.state_dict(), 'python/CV/Resnet50/weights.h5') 

        Resnet50_testModel(model_trained).predictDatasetBySetInput(test_y)

    elif input_str == "test":
        #-----------------------------------    load model    -------------------------------------------
        model = Resnet50_testModel.loadModel('python/CV/Resnet50/weights.h5')

        #-----------------------------------    test model    -------------------------------------------
        testnet = Resnet50_testModel(model)
        #predict = testnet.predictSingleImage('dataset/bad/bad_02.jpg')
        testnet.predictDataset(DIR_PATH)
        


