function dataaugmentation(mainfolder,ymainfolder)
    %% Data augmentation codes 
    % 2020 Dr. Kali Gürkahraman & Dr. Rukiye KARAKIÞ 
    % %Example
    % clc;
    % clear;
    % close all;
    % mainfolder='I:\20 Eylül 2019-Okul Bilgisayar\20 Eylül 2019 -Masaüstü Yedek\DEEPLEARNING\TumorDataset\DATASET\glioma';
    % ymainfolder='I:\20 Eylül 2019-Okul Bilgisayar\20 Eylül 2019 -Masaüstü Yedek\DEEPLEARNING\TumorDataset\DATASET2\glioma';
    % dataaugmentation(mainfolder,ymainfolder);
    %% 
    dfiles=dir(mainfolder);
    dfiles=dfiles(3:end); 
    sm=length(dfiles);
    %%
    for i=1:sm
         %fn=strcat(dfolder(i).folder,'\',dfolder(i).name); 
         filename=strcat(dfiles(i).folder,'\',dfiles(i).name); 
         info=dicominfo(filename);
         image=dicomread(info);  
         %image=imresize(image,[128,128]);% once goruntuyu resize ettim, ayni boyutta olsunlar istiyorum...

         im1 = fliplr(image);           %# horizontal flip
        %figure;imshow(im1,[]);
         im2 = flipud(image);           %# vertical flip

         im3=imrotate(image,-90,'bicubic'); % -90 rotation 
         im4=imrotate(image,-45,'bicubic'); % -45 rotation        
         im5=imrotate(image,45,'bicubic');  % 45 rotation
         im6=imrotate(image,90,'bicubic');  % 90 rotation
         im7= imgaussfilt(image,0.5); % gaussion blur sigma=0.25/0/5/1.5/2
         im8= imgaussfilt(image,1.0); % gaussion blur sigma=0.25/0/5/1.5/2   
         im9= imsharpen(image,'Radius',2,'Amount',1); % value=0.5/1/1.5/2
         im10= imadjust(image,[],[],1.0); % gamma correction value=1.0

         [filepath,name,ext] = fileparts(filename);

         yfilename=strcat(ymainfolder,'\',name,'.dcm'); %original image
         dicomwrite(image, yfilename,info);	

         yfilename=strcat(ymainfolder,'\',name,'mir1','.dcm'); % left-to-right horizontal flip 
         dicomwrite(im1, yfilename,info);	

         yfilename=strcat(ymainfolder,'\',name,'mir2','.dcm'); % up-to-down vertical flip
         dicomwrite(im2, yfilename,info);	

         yfilename=strcat(ymainfolder,'\',name,'rot1','.dcm'); % -90 rotation 
         dicomwrite(im3, yfilename,info);

         yfilename=strcat(ymainfolder,'\',name,'rot2','.dcm'); % -45 rotation 
         dicomwrite(im4, yfilename,info);

         yfilename=strcat(ymainfolder,'\',name,'rot3','.dcm'); % 45 rotation 
         dicomwrite(im5, yfilename,info);

         yfilename=strcat(ymainfolder,'\',name,'rot4','.dcm'); % 90 rotation 
         dicomwrite(im6, yfilename,info);

         yfilename=strcat(ymainfolder,'\',name,'blur1','.dcm'); % 0.50 blurring
         dicomwrite(im7, yfilename,info);

         yfilename=strcat(ymainfolder,'\',name,'blur2','.dcm'); % 1.0 blurring 
         dicomwrite(im8, yfilename,info);

         yfilename=strcat(ymainfolder,'\',name,'sharp1','.dcm'); % sharpening 
         dicomwrite(im9, yfilename,info);

         yfilename=strcat(ymainfolder,'\',name,'gamma1','.dcm'); % -90 rotation 
         dicomwrite(im10, yfilename,info);
    end