# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 11:07:10 2022

@author: oo_wais (Owais Tahir)
"""

import tensorflow as tf
import cv2
import os
import sys
import random
import numpy as np
import glob

from tqdm import tqdm
from patchify import patchify
import tifffile as tif
from PIL import Image
from skimage.io import imread,imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

size_x=128
size_y=128
################# Storing Train Images into an array #############
train_images=[]

for directory_path in glob.glob("Dataset_BUSI_with_GT/data_for_training_and_testing/train/images"):
    for img_path in glob.glob(os.path.join(directory_path,"*.png")):
        #print(img_path)
        img=cv2.imread(img_path,cv2.IMREAD_COLOR)
        img=cv2.resize(img,(size_y, size_x))
        train_images.append(img)
        
train_images = np.array(train_images) #converting list to array

################# Storing Train Masks into an array #############
train_masks = []

for directory_path in glob.glob("Dataset_BUSI_with_GT/data_for_training_and_testing/train/masks"):
    for mask_path in glob.glob(os.path.join(directory_path,"*.png")):
        mask=cv2.imread(mask_path,0)
        mask=cv2.resize(mask,(size_y,size_x))
        train_masks.append(mask)
        
train_masks = np.array(train_masks) #converting list to array

########## Renaming variables to traditional  names ############
X_train = train_images
Y_train = train_masks
Y_train = np.expand_dims(Y_train,axis=3)

########## Displaying random image from X_train and Y_train ######### 
random_num = random.randint(0,623)
imshow(X_train[random_num])
plt.show()
imshow(Y_train[random_num])
plt.show() 


test_img = X_train[random_num]
print(test_img.min(), test_img.max())
print(test_img.shape)

####################### Building Model ############################

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Build U-Net Architecture

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs) #converting inputs from integers to floats

#Contraction/Encoding Path
c1 = tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128,(3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128,(3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(c4)

c5 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive/Decoding Path
u6 = tf.keras.layers.Conv2DTranspose(128,(2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128,(3,3), activation = 'relu', kernel_initializer = 'he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128,(3,3), activation = 'relu', kernel_initializer = 'he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64,(2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', kernel_initializer = 'he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', kernel_initializer = 'he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32,(2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', kernel_initializer = 'he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', kernel_initializer = 'he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16,(2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1], axis=3)
c9 = tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', kernel_initializer = 'he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', kernel_initializer = 'he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1,(1,1), activation = 'sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

######################Training Model####################################
#Model Checkpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_BreastCancerTumorSegmentation.h5', verbose=1, save_best_only=True)

#Early Stopping and Tensorboard
callbacks = [
    #tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]


results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

