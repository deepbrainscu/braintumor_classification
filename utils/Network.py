"""
Created on 2020
@author: Dr. Kali Gurkahraman & Dr. Rukiye KARAKIS
"""
# Base libraries
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard 
from keras.applications import vgg19,vgg16,inception_v3,ResNet50,DenseNet121
from keras.utils.data_utils import get_file
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Input, Dropout, Activation, Flatten, BatchNormalization,ZeroPadding2D,concatenate,Lambda,GlobalAveragePooling2D
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, ZeroPadding3D,GlobalAveragePooling3D

from keras.optimizers import Adam,SGD,RMSprop,Adagrad
from keras import backend as K 
from itertools import product   

from keras import regularizers 


def get_Network(tf_bool,tf_type,num_classes,input_shape,loss, optimizer, metric,height, width,dropbool):
    ## construct CNN structure
    if tf_bool==2: 
        model = Sequential()
        ## 1st convolution layer
        model.add(Conv2D(128, (5, 5), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        #
        ## 2nd convolution layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        #
        ## 3rd convolution layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        #        
        ## 4rd convolution layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        #
        model.add(Flatten())
        #        
        ## fully connected neural networks
        model.add(Dense(1024, activation='relu'))
        if dropbool==1:
            model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        if dropbool==1:
            model.add(Dropout(0.2))
        # Classifier
        model.add(Dense(num_classes, activation='softmax'))          
        # Compile    
        model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
        
    elif tf_bool==1: # TF 
        input_shape=(height, width, 3) 
        if tf_type==1:  # DenseNet 121
            print("TRANSFER OGRENME DenseNet 121 ILE AGI EGITIYORUZ")
            weights = "imagenet"  # "GAN_VGG19_agirliklari.h5"    
            WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
            fname = 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'    
            DenseNet = DenseNet121(weights=weights, include_top=False, input_shape=input_shape)
            DenseNet.load_weights(get_file(fname, WEIGHTS_PATH_NO_TOP, cache_subdir='models'))   
            add_model = Sequential()
            add_model.add(Flatten())
            add_model.add(BatchNormalization())
            add_model.add(Dense(1024, activation='relu'))
            if dropbool==1:
                add_model.add(Dropout(0.3))
            add_model.add(BatchNormalization())
            add_model.add(Dense(512, activation='relu'))
            if dropbool==1:
                add_model.add(Dropout(0.2))
            add_model.add(BatchNormalization())
            add_model.add(Dense(num_classes, activation='softmax'))
    
            model = Model(inputs=DenseNet.input, outputs=add_model(DenseNet.output))
            model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
            print("DenseNet121 network was created successfully") 
    return model
    ## ------------------------------