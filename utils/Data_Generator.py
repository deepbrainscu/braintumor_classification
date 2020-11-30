# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:47:58 2020
@author: rkar
"""
import keras
import numpy as np
import cv2
from keras.preprocessing import image
import numpy as np
from pydicom import dcmread


class Data_Generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(128,128), n_channels=1,n_classes=3, shuffle=False):
          
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.batch_len=self.__len__() # dimension
        self.on_epoch_end() 
        self.bindex = 0
       

    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(np.floor(len(self.list_IDs) / self.batch_size)) 
        return (np.ceil(len(self.list_IDs) / float(self.batch_size))).astype(np.int)       

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch       
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((np.size(list_IDs_temp), *self.dim, self.n_channels))
        y = np.zeros((np.size(list_IDs_temp)), dtype=int)
        # Generate data
        i=0 # counter        
        for file_name in list_IDs_temp:
          gray = dcmread(file_name)
          gray=gray.pixel_array
          gray=cv2.resize(gray,self.dim) # resize            
          if self.n_channels==3: # TF 
              gray = cv2.merge([gray,gray,gray]) 
          if gray.max()!=0:                  
              gray =gray.astype(np.float32) / gray.max() # Normalization 
          gray=image.img_to_array(gray)
          indrk = self.list_IDs.index(file_name)              
          X[i,]=gray         
          y[i] = self.labels[indrk]
          i=i+1
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
  
    def reset(self):
        self.batch_index = 0      
    