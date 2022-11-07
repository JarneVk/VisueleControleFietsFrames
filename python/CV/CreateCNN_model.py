import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CreateCNN_model():
    def __init__(self):
        pass

    def AlexNet(train_x,test_y,epochs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        print("start creating Model:")
        model = AlexNet().to(device)
        print("model created ---> start training model :")

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()

        train_loss = []
        train_acc = []

        #training section

        for epoch in range(epochs):
        
            total_train_loss = 0
            correct = 0
            
            for idx, (image, label) in enumerate(train_x):
                image, label = image.to(device), label.to(device)

                optimizer.zero_grad()
                pred = model(image)
                loss = criterion(pred, label)
                total_train_loss += loss.item()

                for i in range(len(label)):
                    if label[i] == torch.max(pred[i], 0)[1]:
                        correct += 1

                loss.backward()
                optimizer.step()

            total_train_loss = total_train_loss / (idx + 1)
            train_loss.append(total_train_loss)

            total_train_acc = correct/(idx + 1)
            train_acc.append(total_train_acc)

            print(f'Epoch: {epoch} | Train Loss: {total_train_loss}')
            print(f'Epoch: {epoch} | Train Accuracy: {total_train_acc}')


        #test model with test_y
        print("model trained ---> start testing model :")
        total_test_loss = 0
        correct = 0

        for idx, (images, labels) in enumerate(test_y):
            #torch.no_grand() so the gradiant won't be changed now
            with torch.no_grad():
                images, labels = images.to(device), labels.to(device)
                pred = model(images)

                for i in range(len(labels)):
                    if labels[i] == torch.max(pred[i], 0)[1]:
                        correct += 1

        total_test_loss = total_test_loss / (idx + 1)

        total_test_acc = correct/(idx + 1)

        print(f'Test Accuracy: {total_test_acc}')

        return train_loss, train_acc, total_test_loss, total_test_acc

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 2):
        super(AlexNet, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.convolutional(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return torch.softmax(x, 1)   