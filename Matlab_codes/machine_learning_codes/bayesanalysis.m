%function bayesanaliz
clc;
clear;
close all;

load('./data_sagital/tumortest_all.mat'); 
load('./data_sagital/tumortrain_all.mat');  

normalize=1; % normalizasyon
%% Normalizasyon ayarlari
if normalize==1  %% mapstd           
   pveri=mapstd(pveri);     
   %[tegitim,ts] =mapstd(tveri);   
   ptest=mapstd(ptest);              
   %[ttest,ts2] =mapstd(ttest);              
elseif normalize==2     %% mapminmax 
   pveri=mapminmax(pveri);              
   %[tegitim,ts] =mapminmax(tveri);  
   ptest=mapstd(ptest);              
   %[ttest,ts2] =mapstd(ttest);
end

training=pveri;
group=tveri; 
sample=ptest;
[n1,n2]=size(sample);
kayit=0;

% Egitim
nb = fitcnb(training, group);
% Test
cpre = predict(nb,sample);

% figure;plotroc(ttest,cpre);
% figure;plotconfusion(ttest,cpre);
% figure;plotregression(ttest,cpre);

[c_matrixp,Result]=confusion.getMatrix(ttest,cpre);
                    
test_accuracy=Result.Accuracy;
test_error=Result.Error;
test_sensitivity=Result.Sensitivity;
test_specificity=Result.Specificity;
test_precision=Result.Precision;
test_FalsePositiveRate=Result.FalsePositiveRate;
test_F1_score=Result.F1_score;
test_MatthewsCorrelationCoefficient=Result.MatthewsCorrelationCoefficient;
test_Kappa=Result.Kappa;
[testr,testm,testb]= regression(ttest',cpre');
[Xlog,Ylog,Tlog,AUClog] = perfcurve(ttest',cpre',1);
 
%%%%%% Verileri kayit ediyorum...  
testtoplu{kayit+1,1}=testr; % regresyon
testtoplu{kayit+1,2}=test_specificity;  % spectivity
testtoplu{kayit+1,3}=test_sensitivity;  % sensitivity
testtoplu{kayit+1,4}=test_accuracy;  % accuracy degeri
testtoplu{kayit+1,5}=test_MatthewsCorrelationCoefficient;  % ayri regresyon degeri
testtoplu{kayit+1,6}=test_precision;   % precision degeri
testtoplu{kayit+1,7}=test_FalsePositiveRate;   % false positive
testtoplu{kayit+1,8}=test_F1_score;   % false positive        
testtoplu{kayit+1,9}=test_Kappa;   % precision degeri
testtoplu{kayit+1,10}=test_error;   % false positive        
testtoplu{kayit+1,11}=AUClog;   % false positive  

save('bayesanaliz.mat','testtoplu');  
clear testtoplu;
                   