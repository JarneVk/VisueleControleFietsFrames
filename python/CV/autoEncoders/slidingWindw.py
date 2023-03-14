import cv2
import numpy as np
import torch
import torch.nn as nn
import model as AutoEncModel
from torch.autograd import Variable
from torchvision import transforms

IMGSIZE = 80

CHANNELBASE = 32
LATENTDIM = 128

# lossFn = nn.MSELoss()
lossFn = nn.L1Loss()
# lossFn = nn.SmoothL1Loss()

class SlidingWindow():
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncModel.Autoencoder_model3(CHANNELBASE,LATENTDIM,num_input_channels=3).to(self.device)
        self.model.load_state_dict(torch.load('python/CV/autoEncoders/best_weights.h5'))
        self.model.eval()

    def analyse(self,image):
        #cut image
        crops = self.cutImage(image,IMGSIZE,IMGSIZE)
        heatmap = np.zeros(image.shape, dtype=np.uint8) * 255
        
        for i in range(len(crops)):
            currentCrop = np.zeros(image.shape, dtype=np.uint8) * 255
            currentCrop = self.changeCurrentPos(currentCrop,crops[i][1],crops[i][2])
            out1 = cv2.addWeighted(heatmap, 0.90, image, 1, 0)
            output = cv2.addWeighted(currentCrop, 0.90, out1, 1, 0)
            cv2.imshow('map',output)
            cv2.waitKey(1)
            
            #check for fault in neural netwerk
            loader = transforms.Compose([ transforms.ToTensor(),transforms.Resize(IMGSIZE)])
            image_tensor = loader(crops[i][0]).float() 
            image_tensor = Variable(image_tensor, requires_grad=True)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor.cuda()
            img = Variable(image_tensor, requires_grad=True)
            img = img.to(self.device)
            with torch.no_grad():
                deco = self.model(img)
                loss = lossFn(deco, img)

            # print(loss.item())
            if loss.item() > 0.15:
                print(loss.item())
                heatmap = self.addTileToHeatmap(heatmap,crops[i][1],crops[i][2])
        
        output = cv2.addWeighted(heatmap, 0.5, image, 1, 0)
        cv2.imshow('map',output)
        cv2.waitKey()
        cv2.imwrite("python/CV/autoEncoders/heatmap.jpg",output)
        
        

    #@param image = foto die meegegeven wordt
    #@param frameH = hoogte van de deeltjes in pixels
    #@param frameW = breedte van de deeltjes in pixels
    #@output lijst van de deeltjes => [0] foto , [1] linkerbovenhoek, [2] rechter onderhoek
    def cutImage(self,image,frameH,frameW):
        height,width = image.shape[:2]

        h = height/frameH
        w = width/frameW
        anountH = int((h*2)-1)
        anountW = int((w*2)-1) 

        print('H='+str(anountH)+'  W='+str(anountW))
        counter = 0
        crops = []

        offsetH = 0
        offsetW = 0
        for i in range(anountW):
            offsetH = 0
            for j in range(anountH):
                x1=offsetW
                x2=offsetW+frameW
                y1=offsetH
                y2=offsetH+frameH
                offsetH += frameH*0.5
                print(str(counter)+'->'+str(y1)+':'+str(y2)+','+str(x1)+':',str(x2))
                crop = image[int(y1):int(y2),int(x1):int(x2)]
                crplist = [crop,(int(x1),int(y1)),(int(x2),int(y2))]
                crops.append(crplist)
                counter+=1
            offsetW += frameW*0.5 
        #for i in range(len(crops)):
            #print(str(type(crops[i][0]))+' x,y 1='+str(crops[i][1])+'  x,y 2='+str(crops[i][2]))
            #cv2.imshow('crops',crops[i])
            #cv2.waitKey(0)
        return crops
    
    #@param map = heatmap
    #@param ltc = left top corner
    #@param rbc = right botom corner
    #@param kind = good or wrong (0 or 1)
    #@output = heatmap
    def addTileToHeatmap(self,map,ltc,rbc):
        color = (0,0,255) #BGR
        map = cv2.rectangle(map, ltc, rbc, color,cv2.FILLED)
        return map
    
    def addS_TileToHeatmap(self,map,ltc,rbc):
        color = (0,153,255) #BGR
        map = cv2.rectangle(map, ltc, rbc, color,cv2.FILLED)
        return map
    
    def changeCurrentPos(self,map,ltc,rbc):
        map = cv2.rectangle(map, ltc, rbc, (150,50,150),3)
        return map
        

if __name__ == '__main__':
    pict = input('pictureNumber: ')
    im = cv2.imread("python/Camera/out/"+pict+".jpg")
    sliding = SlidingWindow()
    sliding.analyse(im)