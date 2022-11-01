#imports
# import keras
# from keras.datasets import cifar10
# from keras import backend as k
# from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, MaxPooling2D
# from keras.models import Model
# from keras.layers import Concatenate, Dropout,Flatten
# from keras import optimizers,regularizers
# from keras.preprocessing.image import ImageDataGenerator
# from keras.initializers import he_normal
# from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#@param data_dir : het path naar de dataset
def make(data_dir){
    batch_size = 32
    img_height = 180
    img_width = 180

    #load the data set
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
}


