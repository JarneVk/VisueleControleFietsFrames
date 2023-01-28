import cv2
import numpy as np
import os

class makeFrames():
    def __init__(self,path) -> None:
        self.nextfileNumber_bad = 0
        self.nextfileNumber_good = 0
        self.findStartCount()
        print(self.nextfileNumber_bad)
        print(self.nextfileNumber_good)
        self.count = 0
        self.image = cv2.imread(path)
        self.crops = self.cutImage(self.image,50,50)

    def findStartCount(self):
        for i in os.listdir('dataset/bad'):
            self.nextfileNumber_bad += 1
        for i in os.listdir('dataset/good'):
            self.nextfileNumber_good += 1
        self.nextfileNumber_bad += 1
        self.nextfileNumber_good += 1

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
                #print(str(counter)+'->'+str(y1)+':'+str(y2)+','+str(x1)+':',str(x2))
                crop = image[int(y1):int(y2),int(x1):int(x2)]
                crplist = [crop,(int(x1),int(y1)),(int(x2),int(y2))]
                crops.append(crplist)
                counter+=1
            offsetW += frameW*0.5 
        return crops

    def nextframe(self):
        self.count +=1
        out = self.crops[self.count]
        return out
    
    def saveframe(self,kind):
        if kind == 'bad':
            cv2.imwrite('dataset/bad/'+str(self.nextfileNumber_bad)+'.jpg',self.crops[self.count][0])
            print('saved bad:'+str(self.nextfileNumber_bad)+'.jpg')
            self.nextfileNumber_bad +=1
        else:
            cv2.imwrite('dataset/good/'+str(self.nextfileNumber_good)+'.jpg',self.crops[self.count][0])
            print('saved good:'+str(self.nextfileNumber_good)+'.jpg')
            self.nextfileNumber_good +=1
    
    def getOvervieuwImage(self):
        return self.image
    