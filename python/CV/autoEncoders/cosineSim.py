import cv2
import numpy as np
import torch
import sys,os
sys.path.append('python/CV/autoEncoders')
import model_auto as AutoEncModel
from torch.autograd import Variable
from torchvision import transforms
from matplotlib import pyplot as plt
import torch.nn.functional as F
import pandas as pd
import statistics

IMGSIZE = 80

CHANNELBASE = 48
LATENTDIM = 384

DIRPATH_GOOD = "dataset_autoenc/good_dir/good"
DIRPATH_BAD = "dataset_autoenc/bad_dir/bad"

class cosineSim():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncModel.Autoencoder_model3(CHANNELBASE,LATENTDIM,num_input_channels=1).to(self.device)
        self.model.load_state_dict(torch.load('python/CV/autoEncoders/best_weights.h5'))
        self.model.eval()
        self.features_good,self.features_bad = self.createGoodFeatures()

    def createGoodFeatures(self,dirpath=DIRPATH_GOOD):
        features_good  = []
        for image_path in os.listdir(dirpath):
            image = cv2.imread(os.path.join(dirpath,image_path))
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            loader = transforms.Compose([ transforms.ToTensor(),transforms.Resize(IMGSIZE)])
            image_tensor = loader(image).float() 
            image_tensor = Variable(image_tensor, requires_grad=True)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor.cuda()
            img = Variable(image_tensor, requires_grad=True)
            img = img.to(self.device)
            with torch.no_grad():
                eco = self.model.encode(img)
            features_good.append(eco)

        features_bad  = []
        for image_path in os.listdir(DIRPATH_BAD):
            image = cv2.imread(os.path.join(DIRPATH_BAD,image_path))
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            loader = transforms.Compose([ transforms.ToTensor(),transforms.Resize(IMGSIZE)])
            image_tensor = loader(image).float() 
            image_tensor = Variable(image_tensor, requires_grad=True)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor.cuda()
            img = Variable(image_tensor, requires_grad=True)
            img = img.to(self.device)
            with torch.no_grad():
                eco = self.model.encode(img)
            features_bad.append(eco)
        print("features extracted")
        return features_good, features_bad
    
    def calcSimulatrityScore(self,image:cv2.Mat):
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        loader = transforms.Compose([ transforms.ToTensor(),transforms.Resize(IMGSIZE)])
        image_tensor = loader(image).float() 
        image_tensor = Variable(image_tensor, requires_grad=True)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor.cuda()
        img = Variable(image_tensor, requires_grad=True)
        img = img.to(self.device)
        with torch.no_grad():
            img_enc = self.model.encode(img)

        similarity_scores = [F.cosine_similarity(img_enc, x) for x in self.features_good]
        similarity_scores = [x.cpu().detach().item() for x in similarity_scores]
        similarity_scores = [round(x, 3) for x in similarity_scores]

        scores = pd.Series(similarity_scores)
        scores = scores.sort_values(ascending=False)

        sg = scores[scores.index[0]]

        similarity_scores = [F.cosine_similarity(img_enc, x) for x in self.features_bad]
        similarity_scores = [x.cpu().detach().item() for x in similarity_scores]
        similarity_scores = [round(x, 3) for x in similarity_scores]

        scores = pd.Series(similarity_scores)
        scores = scores.sort_values(ascending=False)

        sb = scores[scores.index[0]]
        
        if sg>sb:
            out = "good"
        else:
            out = "bad"

        print(f"score good: {sg} | score bad: {sb} => {out}")

        return out
    
    def test(self):
        good_scores = []
        for im_path in os.listdir("dataset_autoenc/val_dir"):
            im = cv2.imread(os.path.join("dataset_autoenc/val_dir",im_path))
            s = self.calcSimulatrityScore(im)
            good_scores.append(s)

        print(good_scores)

        bad_scores = []
        for idx,im_path in enumerate(os.listdir("dataset_autoenc/bad_dir/bad")):
            if idx >= 111:
                break
            im = cv2.imread(os.path.join("dataset_autoenc/bad_dir/bad",im_path))
            s = self.calcSimulatrityScore(im)
            bad_scores.append(s)

        print("__________________________________")
        print(bad_scores)

        # plt.hist(good_scores, bins=50,label='normal',alpha = 0.5)
        # plt.hist(bad_scores, bins=50, label='anomaly',alpha = 0.5)
        # plt.ylabel('amount')
        # plt.xlabel('score')
        # plt.title('ssim score')
        # plt.legend(loc='upper right')
        # plt.xlim(0,1)
        # plt.show()
        # plt.clf()
        
        fp = 0
        fn = 0
        tp = 0
        for g in good_scores:
            if g == "good":
                continue
            else:
                fp +=1
        
        for b in bad_scores:
            if b == "bad":
                tp +=1
            else:
                fn +=1

        try:
            precission = tp/(tp+fp)
            recal = tp/(tp+fn)
        except ZeroDivisionError:
            precission = 0
            recal = 0

        print(f"Precision : {precission} ; recall : {recal}")
    

if __name__ == "__main__":
    c = cosineSim()
    c.test()