function [f BlkIdx] = TCCANet_FeaExt(InImg,V,CCANet,NumView)
% =======INPUT=============
% InImg     Input images (cell)  
% V         given PCA filter banks (cell)
% PCANet    PCANet parameters (struct)
%       .PCANet.NumStages      
%           the number of stages in PCANet; e.g., 2  
%       .PatchSize
%           the patch size (filter size) for square patches; e.g., [5 3]
%           means patch size equalt to 5 and 3 in the first stage and second stage, respectively 
%       .NumFilters
%           the number of filters in each stage; e.g., [16 8] means 16 and
%           8 filters in the first stage and second stage, respectively
%       .HistBlockSize 
%           the size of each block for local histogram; e.g., [10 10]
%       .BlkOverLapRatio 
%           overlapped block region ratio; e.g., 0 means no overlapped 
%           between blocks, and 0.3 means 30% of blocksize is overlapped 
%       .Pyramid
%           spatial pyramid matching; e.g., [1 2 4], and [] if no Pyramid
%           is applied
% =======OUTPUT============
% f         PCANet features (each column corresponds to feature of each image)
% BlkIdx    index of local block from which the histogram is compuated
% =========================

% addpath('./Utils')

if length(CCANet.NumFilters)~= CCANet.NumStages;
    display('Length(PCANet.NumFilters)~=PCANet.NumStages')
    return
end

NumImg = size(InImg,1);

OutImg = InImg; 
ImgIdx = (1:NumImg)';
clear InImg;
for stage = 1:CCANet.NumStages
     [OutImg ImgIdx] = TCCA_output_test(OutImg, ImgIdx, ...
           CCANet.PatchSize(stage), CCANet.NumFilters(stage), V{stage},NumView);  
end

[f1 BlkIdx] = HashingHist(CCANet,ImgIdx,OutImg{1});
[f2 BlkIdx] = HashingHist(CCANet,ImgIdx,OutImg{2});
[f3 BlkIdx] = HashingHist(CCANet,ImgIdx,OutImg{3});
% f = f1+f2+f3;                                  %FFS1
f = sparse([f1;f2;f3]);                                    %FFS2
% f = sparse([f1;f2]);
% f = sparse(f3);

