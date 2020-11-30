'''
-The Brain Tumor Classification with 2D-T1-weighted MR images 
based Keras and Tensorflow
Created on 2020
@author: Dr. Kali Gurkahraman & Dr. Rukiye KARAKIS
'''
import keras
from keras.models import Sequential, Model, load_model
from keras.models import model_from_json
import json

from keras import backend as K
from keras.applications import DenseNet121
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Input, Dropout, Activation, Flatten, BatchNormalization,ZeroPadding2D,concatenate,Lambda,GlobalAveragePooling2D
from keras.optimizers import Adam,SGD,RMSprop
from keras.utils.data_utils import get_file
from keras.preprocessing import image

import glob
import random
import numpy as np
from skimage import io
import cv2
import pandas as pd

from pydicom import dcmread
from scipy.io import savemat

# load model
json_file = 'D:\makale02\Sonuclar\Sonuclar-dataset6-8 nisan\Sonuclar\sonuc-11\model0.json'
weights_file = 'D:\makale02\Sonuclar\Sonuclar-dataset6-8 nisan\Sonuclar\sonuc-11\sonuc0.h5'
model_json = open(json_file, 'r')
loaded_model_json = model_json.read()
model_json.close()
model = model_from_json(loaded_model_json)
model.load_weights(weights_file)

# variables
n_channels=3
height=128
width=128

# model properties
model.summary()
layer_names=[layer.name for layer in model.layers] # obtain layer names of model
fc1 = Model(inputs=model.input,outputs=model.get_layer(layer_names[424]).output) 

# Train and test dataset filenames 
data_train = pd.read_excel(r'D:\makale02\Tumor_Keras\Tumor_KerasYedek_27Kasım2020\exceltrain.xlsx')
data_test=pd.read_excel(r'D:\makale02\Tumor_Keras\Tumor_KerasYedek_27Kasım2020\exceltest.xlsx')
xdata=[]

# Features of training dataset 
for train_index,train_row in data_train.iterrows():
    # read file and predict 
    gray = dcmread(train_row['id'])
    gray=gray.pixel_array
    gray=cv2.resize(gray,(height,width)) # resize            
    if n_channels==3: # TF 
        gray = cv2.merge([gray,gray,gray])                  
    gray =gray.astype(np.float32) / gray.max() # normalization 
    gray=image.img_to_array(gray)    
    gray = gray.reshape(1, height, width, n_channels) 
     
    # make a prediction
    fc1_output = fc1.predict(gray)
    fc1_output=fc1_output.reshape(fc1_output.shape[0],fc1_output.shape[1]*fc1_output.shape[2]*fc1_output.shape[3])
    xdata.append(fc1_output)
    print(train_index, train_row['id'], train_row['label'])

# save features in a *.mat file
savemat("mat_train.mat", {'train':np.array(xdata)})

# Features of testing dataset 
for test_index,test_row in data_test.iterrows():
    # read file and predict 
    gray = dcmread(test_row['id'])
    gray=gray.pixel_array
    gray=cv2.resize(gray,(height,width)) # resize            
    if n_channels==3: # TF 
        gray = cv2.merge([gray,gray,gray])                  
    gray =gray.astype(np.float32) / gray.max() # normalization 
    gray=image.img_to_array(gray)    
    gray = gray.reshape(1, height, width, n_channels) 
     
    # make a prediction
    fc1_output = fc1.predict(gray)
    fc1_output=fc1_output.reshape(fc1_output.shape[0],fc1_output.shape[1]*fc1_output.shape[2]*fc1_output.shape[3])
    xdata.append(fc1_output)
    print(test_index, test_row['id'], test_row['label'])

# save features in a *.mat file
savemat("mat_test.mat", {'test':np.array(xdata)}) 



