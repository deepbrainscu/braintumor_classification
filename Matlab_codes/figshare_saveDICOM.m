function figshare_saveDICOM(foldername,glioma_folder,meningioma_folder,pituitary_folder)
    %% Figshare mat files to DICOM files--------------------------------------
    % 2020 Dr. Kali Gürkahraman & Dr. Rukiye KARAKIÞ 
    % %Example
    % clc;
    % clear;
    % close all;
    % foldername1='I:\20 Eylül 2019-Okul Bilgisayar\20 Eylül 2019 -Masaüstü Yedek\DEEPLEARNING\TumorDataset\AsilVeriSeti\brainTumorDataPublic_1-766';
    % foldername2='I:\20 Eylül 2019-Okul Bilgisayar\20 Eylül 2019 -Masaüstü Yedek\DEEPLEARNING\TumorDataset\AsilVeriSeti\brainTumorDataPublic_767-1532';
    % foldername3='I:\20 Eylül 2019-Okul Bilgisayar\20 Eylül 2019 -Masaüstü Yedek\DEEPLEARNING\TumorDataset\AsilVeriSeti\brainTumorDataPublic_1533-2298';
    % foldername4='I:\20 Eylül 2019-Okul Bilgisayar\20 Eylül 2019 -Masaüstü Yedek\DEEPLEARNING\TumorDataset\AsilVeriSeti\brainTumorDataPublic_2299-3064';
    % glioma_folder='I:\20 Eylül 2019-Okul Bilgisayar\20 Eylül 2019 -Masaüstü Yedek\DEEPLEARNING\TumorDataset\DATASET\glioma';
    % meningioma_folder='I:\20 Eylül 2019-Okul Bilgisayar\20 Eylül 2019 -Masaüstü Yedek\DEEPLEARNING\TumorDataset\DATASET\meningioma';
    % pituitary_folder='I:\20 Eylül 2019-Okul Bilgisayar\20 Eylül 2019 -Masaüstü Yedek\DEEPLEARNING\TumorDataset\DATASET\pituitary';
    % figshare_saveDICOM(foldername1,glioma_folder,meningioma_folder,pituitary_folder)

    %% Before create folders as DATASET/meningioma, DATASET/glioma, DATASET/pituitary 
    mfiles=dir(foldername); % find the files in foldername
    mfiles=mfiles(3:end); 
    fn=length(mfiles);

    sayac=1;
    for i=1:fn
        filename=strcat(mfiles(i).folder,'\',mfiles(i).name); 
        file=load(filename);
        cjdata=file.cjdata;
        if cjdata.label==1  % 1-meningioma, 2-glioma, 3-pituitary tumor
            nfilename=strcat(meningioma_folder,'\',int2str(sayac),'.dcm');
            dicomwrite(cjdata.image, nfilename); % write dicom file
        elseif cjdata.label==2 %glioma
            nfilename=strcat(glioma_folder,'\',int2str(sayac),'.dcm');
            dicomwrite(cjdata.image, nfilename); % write dicom file
        elseif cjdata.label==3 %pituitary
            nfilename=strcat(pituitary_folder,'\',int2str(sayac),'.dcm');
            dicomwrite(cjdata.image, nfilename); % write dicom file
        end 
        sayac=sayac+1;    
    end
