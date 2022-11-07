import torch
from torch import nn
import torch.utils.data 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.transforms import transforms, Lambda , ToTensor

import joblib
import matplotlib.pyplot as plt
import os
import pandas as pd

import CreateCNN_model

# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

class CreateDataset():
    def __init__(self) -> None:
        pass

    def LoadDataset(csv_path,dir_path):
        # transfor ToTensor() -> converts images to tensors
        # Lambda -> labels to one-hot encoded tensor
        #dataset = CustomImageDataset(csv_path,dir_path)
        transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
          ])
        batch_size = 64
        dataset = datasets.ImageFolder('test_data_cat_dogs/data', transform=transform)
        print("dataset has a size of : "+str(len(dataset)))
        test_aant = int(len(dataset)*0.75)
        train_aant = len(dataset)-test_aant

        train_x, test_y = torch.utils.data.random_split(dataset,[train_aant,test_aant])

        train_dataloader = DataLoader(train_x, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_y, batch_size=batch_size, shuffle=True)
        print("data has been splitted and loaded")
        return train_dataloader, test_dataloader

if __name__ == '__main__':

    #----------------------------------- creating dataset -------------------------------------------
    CSV_PATH = "test_data_cat_dogs/classes.csv"
    DIR_PATH = "test_data_cat_dogs/Img_dataset"
    train_dataloader, test_dataloader = CreateDataset.LoadDataset(CSV_PATH,DIR_PATH)
    train_loss, train_acc, test_loss, test_acc = CreateCNN_model.CreateCNN_model.AlexNet(train_dataloader,test_dataloader,10)
    plt.plot(train_loss)
    plt.show()


