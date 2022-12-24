import cv2
import numpy as np
import os

class imProcessing():

    def __init__(self) -> None:
        pass

    def removeColor(low,high,image):
        imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, low, high)
        mask = 255-mask #reverse mask
        output = cv2.bitwise_and(image, image, mask=mask)
        return output

    def threshhold(img,rm_img):
        imGray = cv2.cvtColor(rm_img,cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(imGray,10,255,cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(thresh,kernel,iterations = 1)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(erosion,kernel,iterations=3)
        output = cv2.bitwise_and(img, img, mask=mask)
        return output

low_blue = np.array([101, 30, 14])
high_blue = np.array([170, 255, 255])

for filename in os.listdir("python/Camera/for_proces"):
    print(filename)
    im = cv2.imread(os.path.join("python/Camera/for_proces",filename))
    output = imProcessing.removeColor(low_blue,high_blue,im)
    output = imProcessing.threshhold(im,output)
    cv2.imwrite(os.path.join("python/Camera/out",filename),output)
        