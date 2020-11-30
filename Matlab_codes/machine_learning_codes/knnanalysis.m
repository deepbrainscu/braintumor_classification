clc;
clear;
close all;

load('./data_dataset6/tumortest_all.mat'); 
load('./data_dataset6/tumortrain_all.mat'); 

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
sayac=1;
for k=2:1:200
    disp(k);
    knnStruct=fitcknn(pveri,tveri,'NumNeighbors',k);
  
    % TEST ETME ISLEMLERI             
    [label,score] = predict(knnStruct,ptest);
    % ROC degerleri
    % figure;plotroc(ttest,label);
    % figure;plotconfusion(ttest,label);
    % figure;plotregression(ttest,label);

    [c_matrixp,Result]=confusion.getMatrix(ttest,label);

    test_accuracy=Result.Accuracy;
    test_error=Result.Error;
    test_sensitivity=Result.Sensitivity;
    test_specificity=Result.Specificity;
    test_precision=Result.Precision;
    test_FalsePositiveRate=Result.FalsePositiveRate;
    test_F1_score=Result.F1_score;
    test_MatthewsCorrelationCoefficient=Result.MatthewsCorrelationCoefficient;
    test_Kappa=Result.Kappa;
    [testr,testm,testb]= regression(ttest',label');
    [Xlog,Ylog,Tlog,AUClog] = perfcurve(ttest',label',1);

    %%%%%% Verileri kayit ediyorum...  
    testtoplu{sayac,1}=testr; % regresyon
    testtoplu{sayac,2}=test_specificity;  % spectivity
    testtoplu{sayac,3}=test_sensitivity;  % sensitivity
    testtoplu{sayac,4}=test_accuracy;  % accuracy degeri
    testtoplu{sayac,5}=test_MatthewsCorrelationCoefficient;  % ayri regresyon degeri
    testtoplu{sayac,6}=test_precision;   % precision degeri
    testtoplu{sayac,7}=test_FalsePositiveRate;   % false positive
    testtoplu{sayac,8}=test_F1_score;   % false positive        
    testtoplu{sayac,9}=test_Kappa;   % precision degeri
    testtoplu{sayac,10}=test_error;   % false positive        
    testtoplu{sayac,11}=AUClog;   % false positive 
    sayac=sayac+1
end

save('knnanaliz2.mat','testtoplu');  
clear toplusonuclar;


