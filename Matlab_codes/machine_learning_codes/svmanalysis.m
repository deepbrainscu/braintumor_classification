%function svmanaliz
clear;
clc;
close all;

load('./data_sagital/tumortest_all.mat'); 
load('./data_sagital/tumortrain_all.mat'); 

normalize=1; % normalizasyon
kayit=0;
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

for i=1:3           
    %%% SVM EGITIM ISLEMI YAPILIYOR             
    %svmStruct = fitcsvm(pveri,tveri2(:,i),'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
    %svmStruct = fitcsvm(pveri,tveri2(:,i),'Standardize',true,'KernelFunction','Gaussian','KernelScale','auto');
    %svmStruct = fitcsvm(pveri,tveri2(:,i),'Standardize',true,'KernelFunction','Linear','KernelScale','auto');
    svmStruct = fitcsvm(pveri,tveri2(:,i),'Standardize',true,'KernelFunction','Polynomial','PolynomialOrder',2,'KernelScale','auto');
    % farkli yollar ekle....

    % TEST ETME ISLEMLERI             
    [label,score] = predict(svmStruct,ptest);

    [c_matrixp,Result]=confusion.getMatrix(ttest2(:,i),label);                    
    test_accuracy=Result.Accuracy;
    test_error=Result.Error;
    test_sensitivity=Result.Sensitivity;
    test_specificity=Result.Specificity;
    test_precision=Result.Precision;
    test_FalsePositiveRate=Result.FalsePositiveRate;
    test_F1_score=Result.F1_score;
    test_MatthewsCorrelationCoefficient=Result.MatthewsCorrelationCoefficient;
    test_Kappa=Result.Kappa;
    [testr,testm,testb]= regression(ttest2(:,i)',label');
    [Xlog,Ylog,Tlog,AUClog] = perfcurve(ttest2(:,i)',label',1);

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
    kayit=kayit+1;
end
save('svmanaliz4.mat','testtoplu');  
clear testtoplu;