import cv2
import os, shutil
import random

#soorten exententies
ORIGINAL = 1
ORIENTATIE = 1
MIRROR = 1

PERSENTAGE = 50

def processing(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(img,(3,3),0)
    ret,threshhold = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshhold

def extendDataset(new_dir:str, old_dir:str):
    shutil.rmtree(new_dir)
    os.makedirs(new_dir)
    os.makedirs(os.path.join(new_dir,"good"))
    os.makedirs(os.path.join(new_dir,"bad"))
    
    for maps in os.listdir(old_dir):
        count = 0
        for file in os.listdir(os.path.join(old_dir,maps)):
            count+=1
        lijst = [0]*count
        for i in range(int(count*PERSENTAGE/100)):
            next =0
            while(next==0):
                ran = int(random.random()*count)
                if lijst[ran] == 0:
                    lijst[ran]=1
                    next =1
            
        x = 0
        for file in os.listdir(os.path.join(old_dir,maps)):
            #copy original files
            if ORIGINAL == 1:
                image = cv2.imread(os.path.join(old_dir,maps,file))
                im_out=processing(image)
                cv2.imwrite(os.path.join(new_dir,maps,file),im_out)
            if lijst[x] ==1 and maps=="bad":
                #roteer willekeurig aantal * 90gr
                if x%2 != 0:
                    if ORIENTATIE ==1:
                        image = cv2.imread(os.path.join(old_dir,maps,file))
                        aant = int(random.random()*3)+1
                        for i in range(aant):
                            rot = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
                        name = file.split(".")[0]+"_rot.jpg"
                        im_out=processing(rot)
                        cv2.imwrite(os.path.join(new_dir,maps,name),im_out)
                else:
                #mirror image 
                    if MIRROR==1:
                        image = cv2.imread(os.path.join(old_dir,maps,file))
                        fl = int(random.random()*3)
                        if fl == 2:
                            fl = -1
                        fliped = cv2.flip(image,fl)
                        name = file.split(".")[0]+"_fliped.jpg"
                        im_out=processing(fliped)
                        cv2.imwrite(os.path.join(new_dir,maps,name),im_out)
            x+=1
        print(maps+" processed")

if __name__ == "__main__":
    extendDataset("dataset_2","dataset")