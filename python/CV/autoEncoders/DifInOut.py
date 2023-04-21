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

from skimage.metrics import structural_similarity as compare_ssim

IMGSIZE = 80

CHANNELBASE = 32
LATENTDIM = 128

class DifInOut():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncModel.Autoencoder_model3(CHANNELBASE,LATENTDIM,num_input_channels=1).to(self.device)
        self.model.load_state_dict(torch.load('python/CV/autoEncoders/best_weights.h5'))
        self.model.eval()

    def evalImage(self,image:cv2.Mat):
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        loader = transforms.Compose([ transforms.ToTensor(),transforms.Resize(IMGSIZE)])
        image_tensor = loader(image).float() 
        image_tensor = Variable(image_tensor, requires_grad=True)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor.cuda()
        img = Variable(image_tensor, requires_grad=True)
        img = img.to(self.device)
        with torch.no_grad():
            deco,_ = self.model(img)
        deco = deco.cpu().numpy()
        deco = deco[0].transpose(1, 2, 0)
        return deco
    
    def cosinSimularScore(self,image):
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        loader = transforms.Compose([ transforms.ToTensor(),transforms.Resize(IMGSIZE)])
        image_tensor = loader(image).float() 
        image_tensor = Variable(image_tensor, requires_grad=True)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor.cuda()
        img = Variable(image_tensor, requires_grad=True)
        img = img.to(self.device)
        with torch.no_grad():
            enc,_ = self.model(img)
        with torch.no_grad():
            enc_enc = self.model.encode(enc)
        with torch.no_grad():
            img_enc = self.model.encode(img)
        similarity_score = F.cosine_similarity(enc_enc, img_enc)
        print(similarity_score)
        similarity_scores = similarity_score.cpu().detach().item()
        score = round(similarity_scores, 3)

        return score
    
    def getScoreDif(image1:cv2.Mat, image2:cv2.Mat):
        image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
        img1 = cv2.resize(image1, (IMGSIZE,IMGSIZE), interpolation = cv2.INTER_AREA)
        img2 = cv2.resize(image2, (IMGSIZE,IMGSIZE), interpolation = cv2.INTER_AREA)
        # cv2.imshow('comp', img1)
        # cv2.waitKey(1)
        # cv2.imshow('comp2', img2)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        (score, diff) = compare_ssim(img1, img2, full=True)
        diff = (diff * 255).astype("uint8")
        # print("SSIM: {}".format(score))
        # thresh = cv2.threshold(diff, 200, 255,
	    # cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imshow('diff',thresh)
        # cv2.waitKey()

        return score
        

def compareAutoencOutput():
    Dif = DifInOut()
    bad_score = []
    for i,image in enumerate(os.listdir("dataset_autoenc/bad_dir/bad")):
        if i > 150:
            break
        im = cv2.imread(os.path.join("dataset_autoenc/bad_dir/bad",image))
        deco = Dif.evalImage(im)
        s =DifInOut.getScoreDif(im,deco)
        bad_score.append(s)

    print("good")
    good_score = []
    for i,image in enumerate(os.listdir("dataset_autoenc/good_dir/good")):
        if i > 150:
            break
        im = cv2.imread(os.path.join("dataset_autoenc/good_dir/good",image))
        deco = Dif.evalImage(im)
        s= DifInOut.getScoreDif(im,deco)
        good_score.append(s)

    plt.hist(good_score, bins=500,label='normal',alpha = 0.5)
    plt.hist(bad_score, bins=50, label='anomaly',alpha = 0.5)
    plt.ylabel('amount')
    plt.xlabel('score')
    plt.title('ssim score')
    plt.legend(loc='upper right')
    plt.xlim(0,0.01)
    plt.show()
    plt.clf()

    thresh = float(input('give threshhold: '))

    fp = 0
    fn = 0
    tp = 0
    for b in bad_score:
        if b < thresh:
            tp +=1
        else:
            fn +=1

    for g in good_score:
        if g < thresh:
            fp += 1

    try:
        precission = tp/(tp+fp)
        recal = tp/(tp+fn)
    except ZeroDivisionError:
        precission = 0
        recal = 0

    print(f"precision : {precission}  |  recal : {recal}")

def cosineSim():
    Dif = DifInOut()

    print("good")
    good_score = []
    for i,image in enumerate(os.listdir("dataset_autoenc/good_dir/good")):
        if i > 20:
            break
        im = cv2.imread(os.path.join("dataset_autoenc/good_dir/good",image))
        s= Dif.cosinSimularScore(im)
        good_score.append(s)

    bad_score = []
    print("bad")
    for i,image in enumerate(os.listdir("dataset_autoenc/bad_dir/bad")):
        if i > 20:
            break
        im = cv2.imread(os.path.join("dataset_autoenc/bad_dir/bad",image))
        s =Dif.cosinSimularScore(im)
        bad_score.append(s)



    print(good_score)
    print(bad_score)

if __name__ == "__main__":
    # compareAutoencOutput()
    cosineSim()

    