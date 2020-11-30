'''
-The Brain Tumor Classification with 2D-T1-weighted MR images 
based Keras and Tensorflow
Created on 2020
@author: Dr. Kali Gurkahraman & Dr. Rukiye KARAKIS
'''
# import folders
import sys 
sys.path.append("./utils")

from data_handling import *
from Network import *
from metric_utils import *
from Data_Generator import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,LearningRateScheduler,TensorBoard # Erken durdurma için

import multiprocessing
import math
from scipy.io import savemat
from functools import partial

# load and save model
from keras.models import Sequential, Model, load_model
from keras.models import model_from_json
import json

# =============================================================================

# variables-------------------------------------------------------------------
cc=2 # cross validation value
num_classes = 3  # glioma, meningioma, pituitory
tumorclasses = ["glioma", "meningioma", "pituitary"]
#imageclass="sagital"; # axial, coronel, sagital
batch_size = 16 
epochs =50  # 
n_channels=1

tf_bool=1 # 1-Transfer Learning 2-No Transfer Learning
if tf_bool==1 :
    n_channels=3
tf_type=1 # Densenet121
height=128 # 
width=128 # 
input_shape=(height, width, 1) #  No Transfer Learning
input_size=(height, width)
# Compile parameters
optsecim=2
if optsecim==1: # SGD
    optimizer= SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
elif optsecim==2: #RMSprop
    optimizer=RMSprop(lr=0.001, rho=0.9)
elif optsecim==3: # Adam
    optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
   
metric="accuracy" 
loss="categorical_crossentropy"
dropbool=1 # 1-Dropout, 2-No Dropout

# Dataset values
valbool=2# 1-Validation, 2-No Validation 
trainrate=0.7 # training rate
testrate=0.3 # testing rate
valrate=0.0  # validation rate
# ---------------------------------------------------------------------------

# initialize trainset and test set
for ii in range(cc): # cross validation
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

    train_generator = Data_Generator(training_list, ytraining_list, batch_size,input_size,n_channels,num_classes,shuffle=True)         
    step_size_train = train_generator.batch_len 
   
    if valbool==1:
        val_generator = Data_Generator(validation_list, yvalidation_list, batch_size,input_size,n_channels,num_classes,shuffle=False)
        step_size_val = val_generator.batch_len              
           
    test_generator =Data_Generator(prediction_list, yprediction_list, batch_size,input_size,n_channels,num_classes,shuffle=False)
    step_size_test=test_generator.batch_len 
## ------------------------------       
    
    
## TRAIN NETWORK  
    model=get_Network(tf_bool,tf_type,num_classes,input_shape,loss, optimizer, metric,height, width,dropbool)
    if valbool==1: 
        #history = model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=epochs,  batch_size=batch_size)
        history = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=step_size_val, steps_per_epoch=step_size_train, epochs=epochs, use_multiprocessing=True,workers=6) 
    else:
        #history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        history = model.fit_generator(train_generator, steps_per_epoch=step_size_train, epochs=epochs,use_multiprocessing=False,shuffle=False)
        # save model
        fnm="model"+str(ii)+".json"
        model_json = model.to_json()
        with open(fnm, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        fnm1="sonuc"+str(ii)+".h5"
        model.save_weights(fnm1)
        print("Saved model to disk")
    ## ------------------------------     
    train_score = model.evaluate_generator(train_generator, steps=step_size_train, verbose=1,max_queue_size=0, workers=0, use_multiprocessing=False)    
    print("TRAINING EVALUATION COMPLETED")
    test_score = model.evaluate_generator(test_generator, steps=step_size_test, verbose=1,max_queue_size=0, workers=0, use_multiprocessing=False)    
    print("PREDICTION EVALUATION COMPLETED")
 
    val_score=[]
    if valbool==1:
        val_score = model.evaluate_generator(val_generator, steps=step_size_val, pickle_safe=False)    
     
# =====================ROC degerleri======================================
    y_test=keras.utils.to_categorical(yprediction_list, num_classes)
    test_pred= model.predict_generator(test_generator,steps=step_size_test,verbose=1,workers=0, max_queue_size=0, use_multiprocessing=False) # generator kapattim-bakalim ne olacak
    test_pred=test_pred[:len(y_test),:]
    test_pred2=np.argmax(test_pred,axis=1)
    test_pred2 = np.array(test_pred2, 'float32')
    test_pred3 = keras.utils.to_categorical(test_pred2, num_classes)   
    test_result=np.zeros((num_classes, 13)) # 
    test_result=metric_all(ii,train_score, test_score, valbool,val_score,history,num_classes,y_test,test_pred3) 
    #del model 
# =============================================================================