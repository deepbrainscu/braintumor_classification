# Brain Tumors Classification with Deep Learning using Data Augmentation

Medical image classification is the process of separating data into a specified number of classes. In this project, we aimed to classify three different brain tumors (glioma, meningioma and pituitary) using convolutional neural network (CNN) on T1-weighted MR images. The weights were initialized by transferring to CNN from DenseNet121 network, which was previously trained with ImageNet dataset. 

In addition, data augmentation was performed on MR images using affine and pixel-level transformations. 

The features obtained from the first fully connected layer of the trained CNN were also classified by support vector machine (DVM), k nearest neighbor (kNN), and Bayes methods. The Matlab files (*.m) of the machine learning methods are given in the Matlab_codes folder.


We couldn't upload trained weights file because there is a 100MB file upload limit on GitHub. You can download pre-trained weight file from **[HERE]( https://drive.google.com/drive/folders/1UBthw32L_4ZL-Trml9yLqAUzWDpjg-yx?usp=sharing)** 


# Step By Step Usage
      
   1- Download all files and folders.
   
   2- If you want to see the performance of the system, download pre-trained weight from **[HERE]( https://drive.google.com/drive/folders/1UBthw32L_4ZL-Trml9yLqAUzWDpjg-yx?usp=sharing)**

  .Then run that file in ** brain_tumor_prediction.py** by using images in /Dataset_DICOM directory.
   
   3- Run all codes in ** brain_tumor_train.py** to classify brain tumors
   
   4- Run all codes in ** brain_tumor_savetraintestlist.py** to save the image file list for the training and testing of the other machine learning classifiers in Matlab.	
   
   5- Run the required codes in ** brain_tumor_featureextraction.py** to obtain features from the CNN model. 
    
   6- Run the required codes in ** brain_tumor_prediction.py** to predict brain tumor class using the MR image file from Dataset_DICOM directory.

# Built with

    Keras
    Tensorflow

# Requirements

    Python 3.6.0
    Numpy
    Keras 2.2.0
    Tensorflow
    h5py 2.9.0
        

# Dataset

The dataset was obtained from the figshare Brain Tumor Database. More details are available to the link below.

**[More Information About figshare Brain Tumor Database](https://figshare.com/articles/brain_tumor_dataset/1512427)**


# Accuracy

The accuracy values of the developed CNN and CNN-based DVM, kNN, and Bayes classifiers are 0.9860, 0.9979, 0.9907, and 0.8933, respectively. 


