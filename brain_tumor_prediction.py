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
#model.summary()
layer_names=[layer.name for layer in model.layers] # obtain layer names of model
fc1 = Model(inputs=model.input,outputs=model.get_layer(layer_names[424]).output) 

# test the CNN model 
filename = r'D:\makale02\Tumor_Keras\Tumor_KerasYedek_27Kasım2020\Dataset_DICOM\Pituitary\713.dcm'

xdata=[]

# Features of training dataset 
gray = dcmread(filename)
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
model_output=model.predict(gray)
model_output2=np.argmax(model_output,axis=1)
output=''
if model_output2[0]==0:
    output='Glioma'
elif model_output2[0]==1:
    output='Meningioma'
elif model_output2[0]==2:
    output='Pituitary'

print("File name=", filename, " \nBrain Tumor Class:",output)

# save features in a *.mat file
savemat("mat_prediction.mat", {'prediction':np.array(xdata)})

