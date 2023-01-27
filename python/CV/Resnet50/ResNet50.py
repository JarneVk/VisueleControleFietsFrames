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

class CreateDataset():
    def __init__(self) -> None:
        pass

    def LoadDataset(dir_path):
        transform = transforms.Compose([
          transforms.Resize((30,30)),
          transforms.ToTensor()
          ])
        batch_size = 8
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
        return train_dataloader, test_dataloader,len(train_x),len(test_y),test_y

class Resnet50_CreateModel():
    def __init__(self,train_dataloader, test_dataloader,num_epochs=3):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.traindata = train_dataloader
        self.testdata = test_dataloader
        self.num_epochs = num_epochs
        self.classmap = joblib.load('python/CV/Resnet50/classmapping')

    def train(self,tsize,valsize):
        model = models.resnet50(weights='DEFAULT').to(self.device)                                           # for pretrained weights='DEFAULT'
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters())
        model_trained = self.train_model(model, criterion, optimizer,tsize,valsize, self.num_epochs)
        return model_trained

    def train_model(self,model, criterion, optimizer,tsize,valsize,num_epochs):
        print(f"train :{tsize} , test :{valsize}")
        t_start = time.time()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
            start_epoch = time.time()
            for phase in ['train', 'validation']:
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
                    print(f"correct counter :{running_corrects} of the {tsize}")
                    epoch_acc = running_corrects.double() / float(tsize)
                else:
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
                    print(f"correct counter :{running_corrects} of the {valsize}")
                    epoch_acc = running_corrects.double() / valsize 

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
                tijd = time.time() - start_epoch
                print(f"time :{tijd}")

        t_tijd = time.time() - t_start
        print(f"total time : {t_tijd}")
        return model


class Resnet50_testModel():
    def __init__(self,model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.mapping = joblib.load('python/CV/Resnet50/classmapping')
        self.transform = transforms.Compose([
          transforms.Resize((30,30)),
          transforms.ToTensor()
          ])

    def loadModel(path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.resnet50(weights=None).to(device)
        model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)
        model.load_state_dict(torch.load(path))
        return model

    def imageLoader(self,input):
        imsize = 30
        loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
        image = Image.open(input)
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        return image.cuda()
    
    def predictSingleImage(self,input):
        im_tensor = self.imageLoader(input)
        predict = self.predict(im_tensor)
        print(f"predicted:{predict}")

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
        train_dataloader, test_dataloader,tsize,valsize,test_y = CreateDataset.LoadDataset(DIR_PATH)
        #-----------------------------------   train model    -------------------------------------------
        resnet = Resnet50_CreateModel(train_dataloader, test_dataloader,20)
        model_trained = resnet.train(tsize,valsize)
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
        


