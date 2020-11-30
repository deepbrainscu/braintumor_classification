"""
Created on 2020
In this class, performance metrics of classes are calculated
@author: Dr. Kali Gurkahraman & Dr. Rukiye KARAKIS 

"""
# Other libraries
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import glob
import random
import numpy as np
from skimage import io
import cv2
import pandas as pd 
import seaborn as sns

# Save the accuracy result in an Excel file
import xlwt 


## The accuracy metrics
from sklearn.metrics import mean_squared_error,  mean_absolute_error, mean_squared_log_error, max_error, explained_variance_score, r2_score
from sklearn.metrics import auc, roc_curve, confusion_matrix,precision_score,recall_score,f1_score,matthews_corrcoef
from math import sqrt


def mcc(tp, fp, tn, fn):
    sen,spec,acc,reg=0,0,0,0
    if tp!=0:
        sen=tp/(tp+fn)
    if tn!=0:
        spec=tn/(fp+tn)
    if (tp+tn)!=0:
        acc=(tp+tn)/(tp+tn+fp+fn)    
    x = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    diff=(tp * tn)-(fp * fn)
    if diff!=0:
        reg=diff/ sqrt(x)
    return sen,spec,acc,reg
   
def score_function(y_test,yfit):
    rkvalue = r2_score(y_test, yfit)
    precision = precision_score(y_test,yfit,zero_division=1, average='binary')
    recall = recall_score(y_test,yfit,zero_division=1, average='binary')
    f1 = f1_score(y_test,yfit,zero_division=1, average='binary')
	#mcc1 = matthews_corrcoef(y_test,yfit)    
    return precision, recall, f1, rkvalue

def write_sheet(ii,index,value,sheet1): # Write the value in an excel sheet 
    sheet1.write(ii, index, value) 

def write_sheet2(ii,index,value,sheet2): # Write the value in an excel sheet 
    sheet2.write(ii, index, value)     

# ROC   
def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')  
  
# Plot Metrics 
def plot_metrics(ii,train_score,test_score,valbool,val_score,history):
    #----PLOT RESULTS----------------------------------------------------------     
    if len(train_score):
        print('Train loss:', train_score[0],'Train accuracy:', train_score[1])
    if valbool==1:
        print('Val loss:', val_score[0],'Val accuracy:', val_score[1])
    
    if len(test_score):
        print('Test loss:', test_score[0],'Test accuracy:', test_score[1])
        
    plt.plot(history.history['accuracy'])    
    if valbool==1:
        plt.plot(history.history['val_accuracy'])        
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left') 
   
    plt.savefig("acc"+str(ii)+".png", bbox_inches='tight', dpi=300)
    plt.savefig("acc"+str(ii)+".pdf", bbox_inches='tight', dpi=300)
    plt.show() 
    
    # summarize history for loss
    plt.plot(history.history['loss'])    
    if valbool==1:
        plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    plt.savefig("loss"+str(ii)+".png", bbox_inches='tight', dpi=300)
    plt.savefig("loss"+str(ii)+".pdf", bbox_inches='tight', dpi=300)
    plt.show()
    
 
def metric_all(ii,train_score, test_score, valbool,val_score,history,num_classes,y_test,test_pred):
    
    # PLOT METRICS
    plot_metrics(ii,train_score,test_score,valbool,val_score,history)    
    workbook = xlwt.Workbook() 
    sheet1 = workbook.add_sheet("RESULTS") 
    sheet2=workbook.add_sheet("R2")
    
    if len(train_score):
        write_sheet(ii, 0, train_score[0],sheet1) 
        write_sheet(ii, 1, 100*train_score[1],sheet1) 
    if len(test_score):
        write_sheet(ii, 2, test_score[0],sheet1) 
        write_sheet(ii, 3, 100*test_score[1],sheet1)
    if not valbool==2:
        write_sheet(ii, 4, val_score[0],sheet1) 
        write_sheet(ii, 5, 100*val_score[1],sheet1)
        
    # ROC statistics    
    test_result=np.zeros((num_classes, 13)) # 
    test_result_mean=np.zeros((1, 14)) # 
    for i in range(num_classes):
        fpr, tpr, thresholds = roc_curve(y_test[:,i], test_pred[:,i])
        auc_t=auc(fpr, tpr)
        tn, fp, fn, tp =confusion_matrix(y_test[:,i], test_pred[:,i],labels=[0,1]).ravel()
        sen_test,spec_test,acc_test,reg=mcc(tp, fp, tn, fn)
        precision, recall,f1,r2=score_function(y_test[:,i], test_pred[:,i])
        test_result[i][0]=auc_t  # area under of curve-AUC
        test_result[i][1]=sen_test # sensitivity
        test_result[i][2]=spec_test #spectivity
        test_result[i][3]=acc_test #accuracy
        test_result[i][4]=reg #regression-correlation coefficient
        test_result[i][5]=precision # precision
        test_result[i][6]=recall #recall
        test_result[i][7]=f1 #f1
        test_result[i][8]=r2 #regression-correlation coefficient  
        test_result[i][9]=tn
        test_result[i][10]=fp
        test_result[i][11]=fn
        test_result[i][12]=tp
       
    test_result_mean=np.mean(test_result, axis=0) # average value of classes
    test_result_mean=np.reshape(test_result_mean, (1, 13))
    
    nk=0    
    for rk in range (6,19):
        write_sheet(ii, rk, test_result_mean[0][nk],sheet1) 
        nk=nk+1        
    
    if ii==0:
        value=0
    else:
        value=((ii+1)*3)-2
        
    for rk in range(0,num_classes):       
        for nk in range (0,13):
            write_sheet2(value, nk, test_result[rk][nk],sheet2) 
        value=value+1 
    excel_name="BRAIN_TUMOUR_RESULT"+str(ii)+".xls"
    workbook.save(excel_name) # Save the result
   # workbook.close(excel_name) # Save the result
    return test_result 