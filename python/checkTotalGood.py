import cv2
import os
from pathlib import Path

from dataclasses import dataclass

IMGSIZE = 80

@dataclass
class Point():
    x : int
    y : int

def do_overlap(l1, r1, l2, r2):
    return not (r1.x < l2.x or l1.x > r2.x or r1.y < l2.y or l1.y > r2.y)


def cutImage(image,frameH,frameW):
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

def checkAnnotation(pic_dir:str):
    img = cv2.imread(pic_dir)
    dh, dw, _ = img.shape

    file_dir = pic_dir[:-3] + 'txt'
    fl = open(file_dir, 'r')
    data = fl.readlines()
    fl.close()

    defectList = []
    for dt in data:

        label, x, y, w, h = map(float, dt.split(' '))

        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        lt = Point(l,t)
        rb = Point(r,b)
        defectList.append((label,lt,rb))
        # cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)
    return defectList

def AnalyseImage(image_path):

    image = cv2.imread(str(image_path))
    bads = 0
    goods = 0

    anottatioList = checkAnnotation(str(image_path))
    crops = cutImage(image,IMGSIZE,IMGSIZE)
    for i in range(len(crops)):
        cropCoord1 = Point(crops[i][1][0],crops[i][1][1])
        cropCoord2 = Point(crops[i][2][0],crops[i][2][1])

        for idx,a in enumerate(anottatioList):
            if do_overlap(l1=cropCoord1,r1=cropCoord2,l2=a[1],r2=a[2]):
                bads += 1

            else:
                goods +=1

    return goods,bads





def main():
    images = Path('test_accuracy').glob('*.jpg')
    totGood = 0
    totBads = 0
    for pic in images:
        g,b = AnalyseImage(pic)

        totGood += g
        totBads += b

    print(f"goods : {totGood}")
    print(f"bads : {totBads}")


if __name__ == "__main__":
    main()