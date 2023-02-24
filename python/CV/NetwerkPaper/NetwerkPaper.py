import os
import time
import joblib, pandas
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import visualise


VALDATA = 'ValidataionSet'
LEARNING_RATE = 0.01

class CreateDataset():
    def __init__(self) -> None:
        pass

    def LoadDataset(dir_path):
        transform = transforms.Compose([
          transforms.Resize((256,256)),
          transforms.ToTensor()
          ])
        batch_size = 32
        dataset = datasets.ImageFolder(dir_path, transform=transform)
        classmapping = dataset.class_to_idx
        print("dataset has a size of : "+str(len(dataset)))
        train_aant = int(len(dataset)*0.80)
        test_aant = len(dataset)-train_aant

        train_x, test_y = torch.utils.data.random_split(dataset,[train_aant,test_aant])

        train_dataloader = DataLoader(train_x, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_y, batch_size=batch_size, shuffle=True)
        print(f"data has been splitted ---- train :{len(train_x)} , test :{len(test_y)}")
        joblib.dump(classmapping,'python/CV/NetwerkPaper/classmapping')

        valdataset = datasets.ImageFolder(VALDATA, transform=transform)
        val_dataloader = DataLoader(valdataset, batch_size=batch_size)

        return train_dataloader, test_dataloader,len(train_x),len(test_y),test_y,val_dataloader,len(valdataset)
    


class LeNet(nn.Module):
    def __init__(self, numChannels, classes):
        # call the parent constructor
        super(LeNet, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20,kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
                    
    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output
    
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
            nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=200,out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=256,out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64,out_features=classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.convolutional(x)
        flat = torch.flatten(conv,1)
        linear = self.linear(flat)
        return linear

class CNN_train():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self,epochs,dataset_train,datase_test,t_size,valsize,val_dataloader,size_val):

        tr_loss = []
        te_loss = []
        val_loss = []

        print("[INFO] initializing the model...")
        # model = LeNet(
	    #         numChannels=3,
	    #         classes=2).to(self.device)
        # model = PaperNet(
	    #         numChannels=3,
	    #         classes=2).to(self.device)
        
        model = models.resnet50(weights='DEFAULT').to(self.device)                                           # for pretrained weights='DEFAULT'
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(self.device)

        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        lossFn = nn.NLLLoss()
        
        for e in range(0, epochs):
            print('Epoch {}/{}'.format(e+1, epochs))
            print('-' * 10)
            model.train()
            totalTrainLoss = 0
            totalValLoss = 0
            trainCorrect = 0
            valCorrect = 0
            i = 0
            # loop over the training set
            for (x, y) in dataset_train:
                (x, y) = (x.to(self.device), y.to(self.device))
                # perform a forward pass and calculate the training loss
                pred = model(x)
                loss = lossFn(pred, y)
                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                opt.zero_grad()
                loss.backward()
                opt.step()

                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            print("Training : Loss: {}".format(totalTrainLoss/t_size))
            print("           Accu: {}".format(trainCorrect/t_size))
            tr_loss.append(totalTrainLoss/t_size)
            
            print('-' * 20)
            # switch off autograd for evaluation
            mapping = joblib.load('python/CV/NetwerkPaper/classmapping')
            rp=0
            fp=0
            fn=0
            rn = 0
            with torch.no_grad():
                model.eval()
                for (x, y) in datase_test:
                    # send the input to the device
                    labels = y
                    (x, y) = (x.to(self.device), y.to(self.device))

                    pred = model(x)

                    for i in range(0,len(y)):
                        expected = list(mapping)[labels[i]]
                        pred_idx = pred[i].argmax(0)
                        prediction = list(mapping)[pred_idx.item()]
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

                    totalValLoss += lossFn(pred, y)
                    valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

            print("Testing: Loss: {}".format(totalValLoss/valsize))
            print("         Accu: {}".format(valCorrect/valsize))
            te_loss.append(totalValLoss/valsize)
            data = [[rp,fp],[fn,rn]]
            headers=["good", "bad"]
            print("_____|expected   size:{}".format(size_val))
            print("pred |")
            print(pandas.DataFrame(data, headers, headers))

            rp=0
            fp=0
            fn=0
            rn = 0

            ValiLoss = 0
            Valicor = 0
            with torch.no_grad():
                model.eval()
                for (x, y) in val_dataloader:
                    # send the input to the device
                    labels = y
                    (x, y) = (x.to(self.device), y.to(self.device))

                    pred = model(x)

                    for i in range(0,len(y)):
                        expected = list(mapping)[labels[i]]
                        pred_idx = pred[i].argmax(0)
                        prediction = list(mapping)[pred_idx.item()]
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
                    ValiLoss += lossFn(pred, y)
                    Valicor += (pred.argmax(1) == y).type(torch.float).sum().item()

            print("Validation:  Loss: {}".format(ValiLoss/size_val))
            print("             Accu: {}".format(Valicor/size_val))
            val_loss.append(ValiLoss/size_val)
            data = [[rp,fp],[fn,rn]]
            headers=["good", "bad"]
            print("_____|expected   size:{}".format(size_val))
            print("pred |")
            print(pandas.DataFrame(data, headers, headers))

        loss_list = [tr_loss,te_loss,val_loss]
        title = "loss:  epochs "+str(epochs)+" | lr "+str(LEARNING_RATE)
        for _ in os.listdir('python/CV/Resnet50/trainingLog'):
            x+=1
        saveLogPict = 'python/CV/NetwerkPaper/trainingLog/log'+str(x)+'.png'
        visualise.Visualise(loss_list,title=title,Save_path=saveLogPict)

        




if __name__ == '__main__':
    DIR_PATH = "dataset_2"
    input_str = input("test or train? :")
    if input_str == "train":
        #----------------------------------- creating dataset -------------------------------------------
        train_dataloader, test_dataloader,tsize,valsize,test_y,val_dataloader,size_val = CreateDataset.LoadDataset(DIR_PATH)
        #-----------------------------------   train model    -------------------------------------------
        epochs = int(input('amount of epochs :'))
        CNN_train().train(epochs,train_dataloader,test_dataloader,tsize,valsize,val_dataloader,size_val)