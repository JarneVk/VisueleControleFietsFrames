from SlidingWindow import SlidingWindow as sl_resnet
from CV.autoEncoders import SlidingWindow_subtract as sl_auto
from CV.autoEncoders import slidingWindw as sl_auto_cossim
from CV.Faster_rcnn.src import detect_image

import cv2
import os, time
from pathlib import Path

from dataclasses import dataclass

RESNET = True
AUTO = True
AUTO_COSIN = False
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
    
    for idx,i in enumerate(detected):
        if i == 0 and int(annotationList[idx][0]) == 0:
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
        if a[0] == 0:       #groot defect
            cv2.rectangle(heatmap, (a[1].x, a[1].y), (a[2].x, a[2].y), (255, 0, 0), 1)
        else:               #small defect
            cv2.rectangle(heatmap, (a[1].x, a[1].y), (a[2].x, a[2].y), (255, 255, 0), 1)
        
    cv2.imshow('img',heatmap)
    cv2.waitKey(500)
    return heatmap

def showFasterResult(image, annotations, boundingboxes):
    for a in annotations:
        if a[0] == 0:       #groot defect
            cv2.rectangle(image, (a[1].x, a[1].y), (a[2].x, a[2].y), (255, 0, 0), 1)
        else:               #small defect
            cv2.rectangle(image, (a[1].x, a[1].y), (a[2].x, a[2].y), (255, 255, 0), 1)
          
    for b in boundingboxes:
        cv2.rectangle(image,(int(b[0][0]), int(b[0][1])),
                            (int(b[1][0]), int(b[1][1])),
                            (0, 0, 255), 2)

    cv2.imshow('img',image)
    cv2.waitKey(500)
    return image


def main(dirpath:str):
    
    precision_resnet_tot = 0
    recal_resnet_tot = 0
    precision_auto_tot = 0
    recal_auto_tot = 0
    precision_auto_tot_cos = 0
    recal_auto_tot_cos = 0
    precision_faster_tot = 0
    recal_faster_tot = 0

    time_resnet = 0
    time_auto = 0
    time_faster = 0

    resnet_tot_fp = 0
    resnet_tot_tp = 0

    #get pictures in dir
    images = Path(dirpath).glob('*.jpg')
    slAuto = sl_auto.SlidingWindow()
    fasterRCNN = detect_image.detect_image()
    slAuto_cosim = sl_auto_cossim.SlidingWindow()
    count = 0
    for pic in images:

        anottatioList = checkAnnotation(str(pic))
        BigDefect = False
        for a in anottatioList:
            if a[0] == 0:
                BigDefect = True

        cv_pic = cv2.imread(str(pic))

        if RESNET == True:
            begin = time.time()
            defectTiles_resnet,heatmap_resnet = sl_resnet.SlidingWindow.analyse(cv_pic)
            cv2.destroyAllWindows()
            time_resnet += time.time() - begin
            fp,fn,tp = calsScore(anottatioList,defectTiles_resnet,heatmap=heatmap_resnet)
            resnet_tot_fp += fp
            resnet_tot_tp += tp
            if BigDefect == False and len(defectTiles_resnet)==0:
                precion_resnet = 1
                recal_resnet = 1
            else:
                precion_resnet, recal_resnet = calcRecal_and_precission(fp,fn,tp)
            print(f"[Resnet50] precision {precion_resnet}  | recal {recal_resnet}")
            out_heatmap = showResult(heatmap_resnet,anottatioList)
            cv2.imwrite("test_accuracy_out/"+str(count)+"_resnet.jpg",out_heatmap)
            precision_resnet_tot += precion_resnet
            recal_resnet_tot += recal_resnet
            cv2.destroyAllWindows()

        if AUTO == True:
            begin = time.time()
            defectTiles_auto,heatmap_auto = slAuto.analyse(cv_pic)
            cv2.destroyAllWindows()
            time_auto += time.time() - begin
            fp,fn,tp = calsScore(anottatioList,defectTiles_auto,heatmap=heatmap_auto)
            if BigDefect == False and len(defectTiles_auto)==0:
                precion_auto = 1
                recal_auto = 1
            else:
                precion_auto, recal_auto = calcRecal_and_precission(fp,fn,tp)
            print(f"[autoEnco] precision {precion_auto}  | recal {recal_auto}")
            out_heatmap = showResult(heatmap_auto,anottatioList)
            cv2.imwrite("test_accuracy_out/"+str(count)+"_auto.jpg",out_heatmap)
            precision_auto_tot += precion_auto
            recal_auto_tot += recal_auto
            cv2.destroyAllWindows()

        if AUTO_COSIN==True:
            ################ cosim ##################################################
            defectTiles_auto,heatmap_auto = slAuto_cosim.analyse(cv_pic)
            cv2.destroyAllWindows()
            fp,fn,tp = calsScore(anottatioList,defectTiles_auto,heatmap=heatmap_auto)
            if len(anottatioList)==0 and len(defectTiles_auto)==0:
                precion_auto = 1
                recal_auto = 1
            else:
                precion_auto, recal_auto = calcRecal_and_precission(fp,fn,tp)
            print(f"[autoEnco Cosinsim] precision {precion_auto}  | recal {recal_auto}")
            out_heatmap = showResult(heatmap_auto,anottatioList)
            cv2.imwrite("test_accuracy_out/"+str(count)+"_auto_cosim.jpg",out_heatmap)
            precision_auto_tot_cos += precion_auto
            recal_auto_tot_cos += recal_auto
            cv2.destroyAllWindows()

        if FASTER == True:
            begin = time.time()
            boxes = fasterRCNN.detect(cv_pic)
            time_faster += time.time() - begin
            fp,fn,tp = calsScore(anottatioList,boxes)
            if BigDefect == False and len(boxes)==0:
                precion_faster = 1
                recal_faster = 1
            else:
                precion_faster, recal_faster = calcRecal_and_precission(fp,fn,tp)    
            print(f"[FasterRcnn] precision {precion_faster}  | recal {recal_faster}")   
            precision_faster_tot += precion_faster
            recal_faster_tot += recal_faster
            out = showFasterResult(cv_pic,anottatioList,boxes)
            cv2.imwrite("test_accuracy_out/"+str(count)+"_faster.jpg",out)
            cv2.destroyAllWindows()



        count +=1

    if RESNET:
        print(f"[total results] Resnet50    : precision {precision_resnet_tot/float(count)} | recal {recal_resnet_tot/float(count)} in {time_resnet} sec")
        print(f"[ResNet info] total fp: {resnet_tot_fp} ; total tp: {resnet_tot_tp}")
    if AUTO:
        print(f"[total results] autoEncoder : precision {precision_auto_tot/float(count)} | recal {recal_auto_tot/float(count)} in {time_auto} sec")
    if AUTO_COSIN:
        print(f"[total results] autoEncoder Cosin sim : precision {precision_auto_tot_cos/float(count)} | recal {recal_auto_tot_cos/float(count)}")
    if FASTER:
        print(f"[total results] fasterRCNN  : precision {precision_faster_tot/float(count)} | recal {recal_faster_tot/float(count)} in {time_faster} sec")

    



if __name__ == '__main__':
    DIRPATH = 'test_accuracy'
    main(DIRPATH)