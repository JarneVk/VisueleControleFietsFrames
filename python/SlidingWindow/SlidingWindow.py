import cv2
import numpy as np
import sys

sys.path.insert(1,"python/CV/Resnet50")
import ResNet50


class SlidingWindow:
    def __init__(self) -> None:
        pass

    def analyse(image):
        #cut image
        crops = SlidingWindow.cutImage(image,50,50)
        heatmap = np.zeros(image.shape, dtype=np.uint8) * 255
        
        for i in range(len(crops)):
            #check for fault in neural netwerk
            model = ResNet50.Resnet50_testModel.loadModel('python/CV/Resnet50/weights.h5')
            testnet = ResNet50.Resnet50_testModel(model)
            cv2.imwrite('python/SlidingWindow/tmp.jpg',crops[i][0])
            predict = testnet.predictSingleImage('python/SlidingWindow/tmp.jpg')
            if predict == "good":
                heatmap = SlidingWindow.addTileToHeatmap(heatmap,crops[i][1],crops[i][2],0)
            if predict == "bad":
                heatmap = SlidingWindow.addTileToHeatmap(heatmap,crops[i][1],crops[i][2],1)

        #opacity on heatmap
        output = cv2.addWeighted(heatmap, 0.4, image, 1 - 0.4, 0)
        cv2.imshow('map',output)
        cv2.waitKey()
        

    #@param image = foto die meegegeven wordt
    #@param frameH = hoogte van de deeltjes in pixels
    #@param frameW = breedte van de deeltjes in pixels
    #@output lijst van de deeltjes => [0] foto , [1] linkerbovenhoek, [2] rechter onderhoek
    def cutImage(image,frameH,frameW):
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
    def addTileToHeatmap(map,ltc,rbc,kind):
        color = (0,0,0)
        if kind == 0:
            color = (0, 255, 0) #in BGR
        else: 
            color = (0, 0, 255)
        print(ltc)
        print(rbc)
        map = cv2.rectangle(map, ltc, rbc, color,cv2.FILLED)
        return map

if __name__ == '__main__':
    im = cv2.imread("python/Camera/out/picture51.jpg")
    SlidingWindow.analyse(im)