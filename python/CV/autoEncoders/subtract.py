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

conf_file = open("python/CV/autoEncoders/NetorkConf.txt","r")
l = conf_file.readline()
conf_file.close()
l = l.strip()
MODEl,CHANNELBASEv,LATENTDIMv = l.split(";")
CHANNELBASEv = int(CHANNELBASEv)
LATENTDIMv = int(LATENTDIMv)
MODEl = int(MODEl)


class subtract():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if MODEl == 4:
            self.model = AutoEncModel.Autoencoder_model4(CHANNELBASEv,LATENTDIMv,num_input_channels=1).to(self.device)
        elif MODEl == 3:
            self.model = AutoEncModel.Autoencoder_model3(CHANNELBASEv,LATENTDIMv,num_input_channels=1).to(self.device)
        self.model.load_state_dict(torch.load('python/CV/autoEncoders/best_weights.h5'))
        self.model.eval()

        self.fileCount=0


    def sub(self,image,thresh=0.18,debug=False):
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        image = cv2.resize(image, (IMGSIZE,IMGSIZE), interpolation = cv2.INTER_AREA)
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
        deco = np.squeeze(deco)


        sub = cv2.subtract(image,deco)
        _,th = cv2.threshold(sub,thresh,1,cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(th,kernel,iterations = 1)
        out = cv2.dilate(erosion,kernel,iterations=1)
        white_pixels = np.sum(out == 1)

        if debug and white_pixels>50:
            # print(white_pixels)
            cv2.imwrite("tmp_out/"+str(self.fileCount)+"_ori.jpg",image*255)
            cv2.imwrite("tmp_out/"+str(self.fileCount)+"_dec.jpg",deco*255)
            cv2.imwrite("tmp_out/"+str(self.fileCount)+"_th.jpg",th*255)
            self.fileCount+=1

        return white_pixels,out
    
def calcRecal_and_precission(fp:int,fn:int,tp:int):
    try:
        precission = tp/(tp+fp)
        recal = tp/(tp+fn)
    except ZeroDivisionError:
        precission = 0
        recal = 0
    return precission,recal

if __name__ == "__main__":
    subtr = subtract()
    bad_tot = []
    good_tot = []
    precission_tot = []
    recall_tot = []
    x_list = []
    x=0.05 
    while x < 0.4:
        good = []
        for i,image in enumerate(os.listdir("dataset_autoenc/good_dir/good")):
            # if i > 200:
            #     break
            im = cv2.imread(os.path.join("dataset_autoenc/good_dir/good",image),cv2.COLOR_BGR2RGB)
            Nw,_ = subtr.sub(im,thresh=x)
            good.append(Nw)
        
        g_mu = np.mean(good)
        g_sigma = np.std(good)
        print(f"Good results {x} : mean {g_mu} ; sigma {g_sigma} => threshold {g_mu+2*g_sigma}")
        good_tot.append(good)

        bad = []
        for i,image in enumerate(os.listdir("dataset_autoenc/bad_dir/bad")):
            # if i > 200:
            #     break
            im = cv2.imread(os.path.join("dataset_autoenc/bad_dir/bad",image),cv2.COLOR_BGR2RGB)
            Nw,_ = subtr.sub(im,thresh=x,debug=False)
            bad.append(Nw)

        mu = np.mean(bad)
        sigma = np.std(bad)
        print(f"Bad results {x} : mean {mu} ; sigma {sigma}")
        bad_tot.append(bad)

        fp = 0
        for g in good:
            if g > (g_mu+2*g_sigma):
                fp +=1

        # print(f"Good : fp {fp}")

        tp = 0
        fn = 0
        for b in bad:
            if b > (g_mu+2*g_sigma):
                tp +=1
            else:
                fn +=1

        # print(f"Bad : tp {tp} ; fn {fn}")

        try:
            precission = tp/(tp+fp)
            recal = tp/(tp+fn)
        except ZeroDivisionError:
            precission = 0
            recal = 0

        print(f"precision {x} : {precission} | recall : {recal}")

        small = []
        for i,image in enumerate(os.listdir("dataset_autoenc/smal_dir/smal")):
            # if i > 200:
            #     break
            im = cv2.imread(os.path.join("dataset_autoenc/smal_dir/smal",image),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            Nw,_ = subtr.sub(im,debug=False)
            small.append(Nw)

        mu = np.mean(small)
        sigma = np.std(small)
        # print(f"Small results : mean {mu} ; sigma {sigma}")    

        tp = 0
        fn = 0
        for b in small:
            if b > (g_mu+2*g_sigma):
                tp +=1
            else:
                fn +=1

        # print(f"Small : tp {tp} ; fn {fn}")

        precisions = []
        recalls = []
        tp=0
        fp=0
        fn=0
        tresh = 0
        while tresh<200:
            for g in good:
                if g>tresh:
                    fp +=1

            for b in bad:
                if b>tresh:
                    tp+=1
                else:
                    fn+=1
            
            precission,recal = calcRecal_and_precission(fp,fn,tp)
            precisions.append(precission)
            recalls.append(recal)
            tresh+=1

        precission_tot.append(precisions)
        recall_tot.append(recalls)

        x_list.append(x)
        x +=0.01

    best = []
    best_idx = (0,0)
    th_idx = []

    for i in range(len(precission_tot)):
        bf1=0
        th = 0
        for idx,p in enumerate(precission_tot[i]):
            f1 = 2*((p*recall_tot[i][idx])/(p+recall_tot[i][idx]))
            if f1 > bf1:
                bf1=f1
                best_ = (p,recall_tot[i][idx])
                th = idx
        
        best.append(best_)
        th_idx.append(th)

    print(best)

    bestPr = 0
    bf1 = 0
    for idx,b in enumerate(best):
        f1 = 2*((b[0]*b[1])/(b[0]+b[1]))
        if f1>bf1:
            bf1 = f1
            bestPr = idx

    print(f"{best[bestPr]} at binaryth {x_list[bestPr]} and th {th_idx[bestPr]}")

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(np_recall_tot, np_precission_tot, Z)
    # ax.set_aspect('equal')
    # plt.show()


    # ax = plt.gca()
    # ax.set_ylim([0.5, 1])
    # ax.set_xlim([0.7, 1])  
    # plt.plot(recalls,precisions,label="precision-recall")
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.title('PR-curve')
    # plt.show()
    