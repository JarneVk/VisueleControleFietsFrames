from SlidingWindow import SlidingWindow as sl_resnet
from CV.autoEncoders import slidingWindw as sl_auto
from CV.Faster_rcnn.src import detect_image

import cv2
import os
from pathlib import Path

from dataclasses import dataclass

RESNET = True
AUTO = True
FASTER = True

@dataclass
class Point():
    x : int
    y : int

def do_overlap(l1, r1, l2, r2):
    return not (r1.x < l2.x or l1.x > r2.x or r1.y < l2.y or l1.y > r2.y)

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
    
def calsScore(annotationList:list[tuple],defectList:list[tuple],heatmap=None):
    fp = 0
    fn = 0
    tp = 0 
    detected = [0] * len(annotationList)
    for d in defectList:

        lt_d = Point(d[0][0],d[0][1])
        rb_d = Point(d[1][0],d[1][1])

        overlap = False
        for idx,a in enumerate(annotationList):
            if do_overlap(l1=lt_d,r1=rb_d,l2=a[1],r2=a[2]):
                tp += 1
                overlap = True
                detected[idx] = 1
                try:
                    if (heatmap).any() != None:
                        cv2.rectangle(heatmap, (d[0][0],d[0][1]), (d[1][0],d[1][1]), (0, 255, 0), 1)
                except AttributeError:
                    pass
        if overlap == False:
            fp += 1
    
    for i in detected:
        if i == 0:
            fn += 1
    
    return fp,fn,tp

def calcRecal_and_precission(fp:int,fn:int,tp:int):
    try:
        precission = tp/(tp+fp)
        recal = tp/(tp+fn)
    except ZeroDivisionError:
        precission = 0
        recal = 0
    return precission,recal
    
def showResult(heatmap,annotations:list[tuple]):

    for a in annotations:
        cv2.rectangle(heatmap, (a[1].x, a[1].y), (a[2].x, a[2].y), (0, 0, 255), 1)
    cv2.imshow('img',heatmap)
    cv2.waitKey(1)


def main(dirpath:str):
    
    precision_resnet_tot = 0
    recal_resnet_tot = 0
    precision_auto_tot = 0
    recal_auto_tot = 0
    precision_faster_tot = 0
    recal_faster_tot = 0

    #get pictures in dir
    images = Path(dirpath).glob('*.jpg')
    slAuto = sl_auto.SlidingWindow()
    fasterRCNN = detect_image.detect_image()
    count = 0
    for pic in images:

        anottatioList = checkAnnotation(str(pic))

        cv_pic = cv2.imread(str(pic))

        if RESNET == True:
            defectTiles_resnet,heatmap_resnet = sl_resnet.SlidingWindow.analyse(cv_pic)
            cv2.destroyAllWindows()
            fp,fn,tp = calsScore(anottatioList,defectTiles_resnet,heatmap=heatmap_resnet)
            precion_resnet, recal_resnet = calcRecal_and_precission(fp,fn,tp)
            print(f"[Resnet50] precision {precion_resnet}  | recal {recal_resnet}")
            showResult(heatmap_resnet,anottatioList)
            precision_resnet_tot += precion_resnet
            recal_resnet_tot += recal_resnet
            cv2.destroyAllWindows()

        if AUTO == True:
            defectTiles_auto,heatmap_auto = slAuto.analyse(cv_pic)
            cv2.destroyAllWindows()
            fp,fn,tp = calsScore(anottatioList,defectTiles_auto,heatmap=heatmap_auto)
            precion_auto, recal_auto = calcRecal_and_precission(fp,fn,tp)
            print(f"[autoEnco] precision {precion_auto}  | recal {recal_auto}")
            showResult(heatmap_auto,anottatioList)
            precision_auto_tot += precion_auto
            recal_auto_tot += recal_auto
            cv2.destroyAllWindows()

        if FASTER == True:
            boxes = fasterRCNN.detect(cv_pic)
            fp,fn,tp = calsScore(anottatioList,boxes)
            precion_faster, recal_faster = calcRecal_and_precission(fp,fn,tp)    
            print(f"[autoEnco] precision {precion_faster}  | recal {recal_faster}")   
            precision_faster_tot += precion_faster
            recal_faster_tot += recal_faster
            cv2.destroyAllWindows()



        count +=1

    print(f"[total results] Resnet50    : precision {precision_resnet_tot/float(count)} | recal {recal_resnet_tot/float(count)}")
    print(f"[total results] autoEncoder : precision {precision_auto_tot/float(count)} | recal {recal_auto_tot/float(count)}")
    print(f"[total results] fasterRCNN  : precision {precision_faster_tot/float(count)} | recal {recal_faster_tot/float(count)}")

    



if __name__ == '__main__':
    DIRPATH = 'test_accuracy'
    main(DIRPATH)