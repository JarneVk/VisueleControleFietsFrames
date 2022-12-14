import numpy as np
import torch
import torch.utils.data 
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import sys

class CreateDataset():
    def __init__(self) -> None:
        pass

    def LoadDataset(dir_path):
        transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
          ])
        batch_size = 64
        dataset = datasets.ImageFolder(dir_path, transform=transform)
        print("dataset has a size of : "+str(len(dataset)))
        train_aant = int(len(dataset)*0.75)
        test_aant = len(dataset)-train_aant

        train_x, test_y = torch.utils.data.random_split(dataset,[train_aant,test_aant])

        train_dataloader = DataLoader(train_x, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_y, batch_size=batch_size, shuffle=True)
        print(f"data has been splitted ---- train :{len(train_x)} , test :{len(test_y)}")
        return train_dataloader, test_dataloader

class Resnet50_CreateModel():
    def __init__(self,train_dataloader, test_dataloader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.traindata = train_dataloader
        self.testdata = test_dataloader
        self.train()

    def train(self):
        model = models.resnet50(weights='DEFAULT').to(self.device)
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters())
        model_trained = self.train_model(model, criterion, optimizer, num_epochs=3)
        return model_trained

    def train_model(self,model, criterion, optimizer,num_epochs=3):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

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

                    epoch_loss = running_loss / (idx + 1)
                    epoch_acc = running_corrects.double() / (idx + 1)
                else:
                    for (inputs, labels) in enumerate(self.testdata):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)




                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
        return model

    def loadModel():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.resnet50(pretrained=False).to(device)
        model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)
        model.load_state_dict(torch.load('models/pytorch/weights.h5'))
        return model

if __name__ == '__main__':
    sys.path.append("/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_cnn_infer.so.8")
    #----------------------------------- creating dataset -------------------------------------------
    DIR_PATH = "dataset"
    train_dataloader, test_dataloader = CreateDataset.LoadDataset(DIR_PATH)
    #-----------------------------------   train model    -------------------------------------------
    model_trained = Resnet50_CreateModel(train_dataloader, test_dataloader)
    #-----------------------------------    save model    -------------------------------------------
    torch.save(model_trained.state_dict(), 'models/pytorch/weights.h5') 
    #-----------------------------------    load model    -------------------------------------------
    #model = Resnet50_CreateModel().loadModel()