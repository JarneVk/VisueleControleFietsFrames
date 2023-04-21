import cv2
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('python/CV/autoEncoders')
import model_auto as AutoEncModel
import subtract
from torch.autograd import Variable
from torchvision import transforms

IMGSIZE = 80

class SlidingWindow():
    def __init__(self) -> None:

        self.sub = subtract.subtract()

    def analyse(self,image):
        defectList = []
        #cut image
        heatmap = np.zeros(image.shape, dtype=np.uint8) * 255

        # image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        # image = cv2.equalizeHist(image)
        # image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        for sizeLoop in range(1,2,1):
            # print(f"kernel size {IMGSIZE/sizeLoop}")
            crops = self.cutImage(image,IMGSIZE/sizeLoop,IMGSIZE/sizeLoop)

            for i in range(len(crops)):
                currentCrop = np.zeros(image.shape, dtype=np.uint8) * 255
                currentCrop = self.changeCurrentPos(currentCrop,crops[i][1],crops[i][2])
                out1 = cv2.addWeighted(heatmap, 0.90, image, 1, 0)
                output = cv2.addWeighted(currentCrop, 0.90, out1, 1, 0)
                cv2.imshow('map',output)
                cv2.waitKey(1)
                score,bin_out = self.sub.sub(crops[i][0])
                if score > 50:
                    # print(score)
                    addTile_r = cv2.cvtColor(bin_out,cv2.COLOR_GRAY2RGB).astype(np.uint8)*255
                    addTile_r = cv2.resize(addTile_r, (int(IMGSIZE/sizeLoop),int(IMGSIZE/sizeLoop)), interpolation = cv2.INTER_AREA)
                    addTile_r[np.where((addTile_r==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
                    # cv2.imshow('t',addTile)
                    # cv2.waitKey()
                    heatmap = self.addTileToHeatmap(heatmap,crops[i][1],crops[i][2],addTile_r)
                    defectList.append((crops[i][1],crops[i][2]))

        
        output = cv2.addWeighted(heatmap, 1, image, 0.5, 0)
        cv2.imwrite("python/CV/autoEncoders/heatmap.jpg",output)

        return defectList,output
        
        

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

        # print('H='+str(anountH)+'  W='+str(anountW))
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
                # print(str(counter)+'->'+str(y1)+':'+str(y2)+','+str(x1)+':',str(x2))
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
    def addTileToHeatmap(self,map,ltc,rbc,add):
        x_offset= ltc[1]
        y_offset= ltc[0]
        # print(map.shape)
        # print(f"{ltc};{rbc}")
        # print(f"{x_offset}:{x_offset+add.shape[1]};{y_offset}:{y_offset+add.shape[0]}")
        # print(f"shape add: {add.shape}")
        map[ x_offset:x_offset+add.shape[1],y_offset:y_offset+add.shape[0]] += add
        color = (0,0,255) #BGR
        map = cv2.rectangle(map, ltc, rbc, color)
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
    im = cv2.imread("python/Camera/out/"+pict+".jpg",cv2.COLOR_BGR2RGB)
    sliding = SlidingWindow()
    sliding.analyse(im)