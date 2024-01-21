%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% cvpr_computedescriptors.m
%% Skeleton code provided as part of the coursework assessment
%% This code will iterate through every image in the MSRCv2 dataset
%% and call a function 'extractRandom' to extract a descriptor from the
%% image.  Currently that function returns just a random vector so should
%% be changed as part of the coursework exercise.
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

close all;
clear all;

%% Edit the following line to the folder you unzipped the MSRCv2 dataset to
DATASET_FOLDER = 'C:\Users\kamil\Surrey_Masters\Computer Vision & Pattern Recognition\Assignment\MSRC_ObjCategImageDatabase_v2';

%% Create a folder to hold the results...
OUT_FOLDER = 'C:\Users\kamil\Surrey_Masters\Computer Vision & Pattern Recognition\Assignment\descriptors';
%% and within that folder, create another folder to hold these descriptors
%% the idea is all your descriptors are in individual folders - within
%% the folder specified as 'OUT_FOLDER'.
%OUT_SUBFOLDER='globalRGBhisto';
OUT_SUBFOLDER= 'SpatialGridDescriptor';
%OUT_SUBFOLDER= 'SpatialGrid_Texture';
%OUT_SUBFOLDER= 'CNN_Descriptor';



allfiles=dir (fullfile([DATASET_FOLDER,'/Images/*.bmp']));
for filenum=1:length(allfiles)
    fname=allfiles(filenum).name;
    fprintf('Processing file %d/%d - %s\n',filenum,length(allfiles),fname);
    tic;
    imgfname_full=([DATASET_FOLDER,'/Images/',fname]);
    img=double(imread(imgfname_full))./256;
    %img=double(imread(imgfname_full));  %for CNN 
    %imageIndex = indexImages(bookCovers);
    fout=[OUT_FOLDER,'/',OUT_SUBFOLDER,'/',fname(1:end-4),'.mat'];%replace .bmp with .mat
    
    descriptor = 'SpatialGridDescriptor';

    switch descriptor
        case 'meanColor'
            F = extractMeanColor(img);
        case 'globalColorHistogram'
            F = ComputeRGBHistogram(img, 5);
        case 'SpatialGridDescriptor'
            F = computeSpatialGridDescriptor(img, 4,4);
        case 'SpatialGrid_Texture'
            F = SpatialGrid_Texture(img, 4,4,4);
        case 'CNN_Encoding'
            F = CNN_Descriptor(img);
        otherwise
            F = extractMeanColor(img);
    end

    save(fout, 'F');

    toc
end
