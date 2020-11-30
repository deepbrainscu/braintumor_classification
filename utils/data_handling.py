"""
Created on Sat May  9 18:42:19 2020

@author: Dr. Rukiye KARAKIS
"""
import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import glob
import random
import numpy as np
from skimage import io
import cv2
import pandas as pd 
from natsort import natsorted
from keras.preprocessing import image


from pydicom import dcmread
from numpy.random import seed # to generate random indexes of patient files

def get_files(strokeclass,trainrate,testrate,valrate,valbool): #Define function to get file list, randomly shuffle it and split 70/30
    files = glob.glob("D:\makale02\Tumor_Keras\Tumor_KerasYedek_27Kasım2020\Dataset_DICOM\{0}\*".format(strokeclass))
#    files = glob.glob("G:\makale02\Tumor_Keras/dataset3\{0}\{1}\*".format(strokeclass, imageclass)) 
    files = shuffle(files)
    if valbool==1: # Validation    
        training = files[:int(len(files)*trainrate)]     
        validation = files[int(len(files)*trainrate+1):int(len(files)*(trainrate+valrate))]
        prediction = files[-int(len(files)-(len(training)+len(validation))):] 
        return training, prediction, validation
    elif valbool==2:# No validation
        training = files[:int(round(len(files)*trainrate))] 
        prediction= files[-int(round(len(files)*testrate)):]      
        validation=[]
        return training, prediction,validation   
    
def make_sets(tf_bool,valbool,tumorclasses,trainrate,testrate,valrate,height,width): 
    x_train, y_train, x_test, y_test, x_val, y_val= [], [], [], [], [], []
    data = {}
    for tclass in tumorclasses:
        training, prediction, validation = get_files(tclass,trainrate,testrate,valrate,valbool)
        #Append data to training and prediction list, and generate class labels [0-2]
        for item in training:
            gray = dcmread(item)
            gray=gray.pixel_array
            gray=cv2.resize(gray,(height,width)) # resize             
            if tf_bool==1: # TF
                gray = cv2.merge([gray,gray,gray])                   
            gray =gray.astype(np.float32) / gray.max() # normalization 
            gray=image.img_to_array(gray)        
            x_train.append(gray) #append image array to training list
            y_train.append(tumorclasses.index(tclass))
        for item in prediction: #repeat above process for prediction set
            gray = dcmread(item)
            gray=gray.pixel_array
            gray=cv2.resize(gray,(height,width)) # resize  
            
            if tf_bool==1: # TF var ise veri 3-boyutlu olmali
                gray = cv2.merge([gray,gray,gray]) 
            gray =gray.astype(np.float32) / gray.max() # normalization
            gray=image.img_to_array(gray)   
            x_test.append(gray)
            y_test.append(tumorclasses.index(tclass))
        if valbool==1: # validasyon varsa    
            for item in validation:
                gray = dcmread(item)
                gray=gray.pixel_array
                gray=cv2.resize(gray,(height,width)) # resize  
                
                if tf_bool==1: # TF
                    gray = cv2.merge([gray,gray,gray]) 
                gray =gray.astype(np.float32) / gray.max() # Normalization
                gray=image.img_to_array(gray)   
                x_val.append(gray)  # append image array to validation list
                y_val.append(tumorclasses.index(tclass))
                
    return x_train, y_train, x_test, y_test, x_val, y_val

def get_datasets(num_classes,tf_bool,valbool,tumorclasses,trainrate,testrate,valrate,height,width):  
    
    x_train, y_train, x_test, y_test, x_val, y_val=make_sets(tf_bool,valbool,tumorclasses,trainrate,testrate,valrate,height,width)
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')

    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')
    
    if valbool==1: # Validation
        y_val = keras.utils.to_categorical(y_val, num_classes)
        x_val = np.array(x_val, 'float32')
        y_val = np.array(y_val, 'float32')
    
    if tf_bool==2: #No TF 
        x_train = x_train.reshape(x_train.shape[0], height, width, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.reshape(x_test.shape[0], height, width, 1)
        x_test = x_test.astype('float32')
        if valbool==1: 
            x_val = x_val.reshape(x_val.shape[0], height, width, 1)
            x_val = x_val.astype('float32')
        
    elif tf_bool==1: # TF
        x_train = x_train.reshape(x_train.shape[0], height, width, 3)
        x_train = x_train.astype('float32')
        x_test = x_test.reshape(x_test.shape[0], height, width, 3)  
        x_test = x_test.astype('float32') 
        
        if valbool==1:
            x_val = x_val.reshape(x_val.shape[0], height, width, 3)
            x_val = x_val.astype('float32')
       
    print(x_train.shape[0], 'training samples')
    print(x_test.shape[0], 'test samples')
    
    if valbool==1:
        print(x_val.shape[0], 'validation samples')    

    return x_train, y_train, x_test, y_test, x_val, y_val