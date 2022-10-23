import cv2
import numpy as np

class imageProcessing():
    
    def __init__(self):
        self.image = cv2.imread('python/Camera/testFiets.jpg')

    def segmentation(self,edges):
        contours,hierarchy = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        mask = np.zeros((256,256), np.uint8)
        masked = cv2.drawContours(mask, contours,-1, 255, -1)
        return masked

    #@param image: give an image in Mat type 
    #@return : returns the edges in Mat type
    def edgeDetection(self,image):
        # cenvert naar greyscale -> blur voor beter echte randen te vinden -> sobel edge detection
        img_grijs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(img_grijs, np.mean(img_grijs), 255, cv2.THRESH_BINARY_INV)
        img_blur = cv2.GaussianBlur(thresh,(3,3),0)
        #img_edge = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        output_img = cv2.Canny(image=img_blur, threshold1=50, threshold2=100)
        cv2.imshow('edges',output_img)
        cv2.waitKey(0)
        output_img = self.segmentation(output_img)
        cv2.imshow('edges',output_img)
        cv2.waitKey(0)
        return output_img

    def cropFrame(self,h,w):
        height,width = self.image.shape[:2]
        frameH = height/h
        frameW = width/w

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
                crop = self.image[int(y1):int(y2),int(x1):int(x2)]
                crops.append(crop)
                counter+=1
            
            offsetW += frameW*0.5
                
                
        for i in range(len(crops)):
            hei,wi = crops[i].shape[:2]
            print(str(type(crops[i]))+' h='+str(hei)+'  w='+str(wi))
            #cv2.imshow('crops',crops[i])
            #cv2.waitKey(0)
        
        return crops

im = imageProcessing()
crops = []
crops = im.cropFrame(3,3)
for i in range(len(crops)):
    crops[i] = im.edgeDetection(crops[i])


