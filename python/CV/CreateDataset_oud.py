import numpy as np
import os
import cv2
import random
import joblib

IMG_SIZE_W = 50 #for our camera 800 is full resolution
IMG_SIZE_H = 50 #for our camera 600 is full resolution

#@param data_dir : path to the main folder that contains the subfolders with categories
#@param classes : subfolders of the main folders that represent the categories
def createDataset(data_dir,classes):
    training_data = []
    for clas in classes:
        path = os.path.join(data_dir,clas)
        class_num = classes.index(clas) #binary representation
        for img in os.listdir(path):
            try:
                im_mat = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) #read in grayscale because it takes in less memory and the images are saved in grayscale
                im_norm = cv2.resize(im_mat,(IMG_SIZE_W,IMG_SIZE_H))
                training_data.append([im_norm,class_num])
            except Exception as e:
                pass
    
    print('length loaded data:'+str(len(training_data)))
    random.shuffle(training_data) #shuffle the data so its not all in the order of the categories

    train_x = []
    test_y = []

    for features,label in training_data:
        train_x.append(features)
        test_y.append(label)
    train_x = np.array(train_x).reshape(-1, IMG_SIZE_W, IMG_SIZE_H, 1) # tf expects a numpy array

    #save the data
    joblib.dump(train_x,"python/CV/train_x.joblib")
    joblib.dump(test_y,"python/CV/test_y.joblib")

                


data_dir = "test_data_cat_dogs/PetImages"
classes = ["Dog","Cat"]

createDataset(data_dir,classes)