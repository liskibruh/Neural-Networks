# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 14:53:22 2022

@author: oo_wa
"""

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread,imshow
import random


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

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
X = train_images
Y = train_masks
Y = np.expand_dims(Y,axis=3)

########## Splitting ####################
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

############# Pre-processing input############
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

########## Displaying random image from X_train and Y_train ######### 
random_num = random.randint(0,400)
imshow(x_train[random_num])
plt.show()
imshow(y_train[random_num])
plt.show() 

####################### Define Model ############################
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


history=model.fit(x_train, 
          y_train,
          batch_size=8, 
          epochs=10,
          verbose=1,
          validation_data=(x_val, y_val))

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
####################

model.save('BUSI_Segmentation.h5')

######################

from tensorflow import keras
model = keras.models.load_model('BUSI_Segmentation.h5', compile=False)
#Test on a different image
#READ EXTERNAL IMAGE...
test_img = cv2.imread('Dataset_BUSI_with_GT/data_for_training_and_testing/test/images/benign (18).png', cv2.IMREAD_COLOR)       
test_img = cv2.resize(test_img, (size_y, size_x))
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
test_img = np.expand_dims(test_img, axis=0)
prediction = model.predict(test_img)

#View and Save segmented image
prediction_image = prediction.reshape(mask.shape)
plt.imshow(prediction_image, cmap='gray')
#plt.imsave('membrane/test0_segmented.jpg', prediction_image, cmap='gray')