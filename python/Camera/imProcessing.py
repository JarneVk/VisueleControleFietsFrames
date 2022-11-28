import cv2
import numpy as np

class imProcessing():

    def __init__(self) -> None:
        pass

    def removeColor(low,high,image):
        imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, low, high)
        mask = 255-mask #reverse mask
        output = cv2.bitwise_and(image, image, mask=mask)
        
        return output

    def threshhold(img):
        img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
        imGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(imGray,10,255,cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(thresh,kernel,iterations = 2)
        dil = cv2.dilate(erosion,kernel,iterations=3)
        kernel = np.ones((3,3),np.uint8)
        output = cv2.erode(dil,kernel,iterations = 1)
        return output

low_blue = np.array([101, 30, 14])
high_blue = np.array([170, 255, 255])

for i in range(0,29,1):
    print('python/Camera/tmp_pict/picture'+str(i)+'.jpg')
    im = cv2.imread('python/Camera/tmp_pict/picture'+str(i)+'.jpg')
    output = imProcessing.removeColor(low_blue,high_blue,im)
    output = imProcessing.threshhold(output)
    cv2.imwrite('python/Camera/tmp_out/picture'+str(i)+'.jpg',output)
        