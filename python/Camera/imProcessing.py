from email.mime import image
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

        cv2.imshow('out',output)
        cv2.waitKey(0)
        
        
        return output

low_blue = np.array([101, 30, 14])
high_blue = np.array([170, 255, 255])
for i in range(0,30,1):
    im = cv2.imread('python/Camera/tmp_pict/picture'+str(i)+'.jpg')
    imProcessing.removeColor(low_blue,high_blue,im)

        