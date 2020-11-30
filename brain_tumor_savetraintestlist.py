'''
-The Brain Tumor Classification with 2D-T1-weighted MR images 
based Keras and Tensorflow
Created on 2020
@author: Dr. Kali Gurkahraman & Dr. Rukiye KARAKIS
'''
import sys 
sys.path.append("./utils")

from data_handling import *
from Network import *
from metric_utils import *
from Data_Generator import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,LearningRateScheduler,TensorBoard # Erken durdurma için

#import cv2
import glob
import random
import numpy as np
from skimage import io
import cv2
import pandas as pd


from numpy.random import randn
from sklearn.utils import shuffle
import numpy as np
import xlwt
import pandas as pd
from xltable import *
# from xlpandas import XLtable, XLseries
from xlwt import Workbook, Worksheet, Style
import os

# =============================================================================
num_classes = 3  # glioma, meningioma, pituitory
# Veri setini bolme adimlari
valbool=2# 1 ise Validasyon var,  2 ise yok
trainrate=0.7 # training rate
testrate=0.3 # testing rate
valrate=0.0  # validation rate
sel_load=2 # 1: normal yükleme, 2 ise generator ile aşamalı yükleme yapacak, 3- keras aşamalı yükleme
tf_bool=1
height=128
width=128
# ------------------------------
# initialize trainset and test set
x_train, y_train, x_test, y_test, x_val, y_val= [], [], [], [], [], []
# transfer train and test set data
tumorclasses = ["glioma", "meningioma", "pituitary"]
#imageclass="sagital"; # axial, coronel, sagital
data = {}
cc=1

training_list, prediction_list, validation_list=[],[],[]
ytraining_list, yprediction_list, yvalidation_list=[],[],[]
for tclass in tumorclasses:
    training, prediction, validation = get_files(tclass,trainrate,testrate,valrate,valbool)
    for item in training:
        training_list.append(item)            
        ytraining_list.append(tumorclasses.index(tclass))  
    for item in prediction: 
        prediction_list.append(item)  
        yprediction_list.append(tumorclasses.index(tclass))
    for item in validation:
        validation_list.append(item)
        yvalidation_list.append(tumorclasses.index(tclass)) 
            
    yprediction_list = list(map(str, yprediction_list))
    yvalidation_list = list(map(str, yvalidation_list))
    ytraining_list = list(map(str, ytraining_list))

    dtrain={'id':training_list,'label':ytraining_list}
    data_dtrain=pd.DataFrame(dtrain,columns=['id', 'label'])
    data_dtrain.to_excel (r'D:\makale02\Tumor_Keras\Tumor_KerasYedek_27Kasım2020\exceltrain.xlsx', index = False, header=True)

    dtest={'id':prediction_list,'label':yprediction_list}
    data_dtest=pd.DataFrame(dtest,columns=['id', 'label'])
    data_dtest.to_excel (r'D:\makale02\Tumor_Keras\Tumor_KerasYedek_27Kasım2020\exceltest.xlsx', index = False, header=True)
# =============================================================================