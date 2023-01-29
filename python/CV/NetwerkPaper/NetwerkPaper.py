import os
import time
import joblib, pandas
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
        train_aant = int(len(dataset)*0.85)
        test_aant = len(dataset)-train_aant

        train_x, test_y = torch.utils.data.random_split(dataset,[train_aant,test_aant])

        train_dataloader = DataLoader(train_x, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_y, batch_size=batch_size, shuffle=True)
        print(f"data has been splitted ---- train :{len(train_x)} , test :{len(test_y)}")
        joblib.dump(classmapping,'python/CV/NetwerkPaper/classmapping')
        return train_dataloader, test_dataloader,len(train_x),len(test_y),test_y

class PaperNet():
    def __init__(self,train_dataloader,test_dataloader) -> None:
        self.traindata = train_dataloader
        self.testdata = test_dataloader
        self.classmap = joblib.load('python/CV/NetwerkPaper/classmapping')

    def train(self,num_epochs,tsize,valsize):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        model = NetwerkPaper().to(device)
        print(f"Model structure: {model}\n\n")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

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
                        inputs = inputs.to(device)
                        labels = labels.to(device)
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
                    for idx, (inputs, labels) in enumerate(self.testdata):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        with torch.no_grad():
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
    
def evalModel(model,dataset):
    mapping = joblib.load('python/CV/NetwerkPaper/classmapping')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rp=0
    fp=0
    fn=0
    rn = 0
    for i in range(len(dataset)):
        img,label = dataset[i]
        img = Variable(img, requires_grad=True)
        img = img.unsqueeze(0)
        image = img.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(image)
            pred_idx = pred[0].argmax(0)
            #print(f"classmap:{self.mapping} -> index: {pred_idx.item()}")
            prediction = list(mapping)[pred_idx.item()]
        expected = list(mapping)[label]
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
    
    



class NetwerkPaper(nn.Module):
    def __init__(self):
        super(NetwerkPaper, self).__init__()
        self.flatten = nn.Flatten()
        self.convolutional_groot = nn.Sequential(
            # 8@5x5 conv layer
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 8@5x5 conv layer
            nn.Conv2d(in_channels=8,out_channels=64,kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 8@3x3 conv layer
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 8@3x3 conv layer
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 8@3x3 conv layer
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.convolutional_klein = nn.Sequential(
            # 8@3x3 conv layer
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 8@3x3 conv layer
            nn.Conv2d(in_channels=8,out_channels=64,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 8@3x3 conv layer
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 8@3x3 conv layer
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 8@3x3 conv layer
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.LazyLinear(64),
            nn.ReLU(inplace=True),
            nn.LazyLinear(2),
            nn.ReLU(inplace=True),
        )

    #flow of the netrwerk with x as input
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.convolutional_klein(x)
        flat = torch.flatten(conv, 1)
        linear = self.linear(flat)
        return torch.softmax(linear, 1) 
    
if __name__ == '__main__':
    DIR_PATH = "dataset_2"
    input_str = input("test or train? :")
    if input_str == "train":
        #----------------------------------- creating dataset -------------------------------------------
        train_dataloader, test_dataloader,tsize,valsize,test_y = CreateDataset.LoadDataset(DIR_PATH)
        #-----------------------------------   train model    -------------------------------------------
        epochs = int(input('amount of epochs :'))
        pn = PaperNet(train_dataloader, test_dataloader)
        model_trained = pn.train(epochs,tsize,valsize)
        #-----------------------------------    save model    -------------------------------------------
        torch.save(model_trained.state_dict(), 'python/CV/NetwerkPaper/weights.h5') 
        evalModel(model_trained,test_y)