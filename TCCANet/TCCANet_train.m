function [f V BlkIdx] = TCCANet_train(InImg,TCCANet,IdtExt,NumView)
% =======INPUT=============
% InImg     Input images (cell); each cell can be either a matrix (Gray) or a 3D tensor (RGB)  
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
% IdtExt    a number in {0,1}; 1 do feature extraction, and 0 otherwise  
% =======OUTPUT============
% f         PCANet features (each column corresponds to feature of each image)
% V         learned PCA filter banks (cell)
% BlkIdx    index of local block from which the histogram is compuated
% =========================

% addpath('./Utils')

if length(TCCANet.NumFilters)~= TCCANet.NumStages;
    display('Length(CCANet.NumFilters)~=CCANet.NumStages')
    return
end

NumImg = length(InImg{1});

V = cell(TCCANet.NumStages,1); %V��2*1 cell
OutImg = InImg; 
ImgIdx = (1:NumImg)';
clear InImg; 

for stage = 1:TCCANet.NumStages
    display(['Computing TCCA filter bank and its outputs at stage ' num2str(stage) '...'])
    tic;
    V{stage} = TCCA_FilterBank_broadcast(OutImg, TCCANet.PatchSize(stage), TCCANet.NumFilters(stage), TCCANet, stage); % compute PCA filter banks
    t=toc;
    if stage ~= TCCANet.NumStages % compute the PCA outputs only when it is NOT the last stage
        [OutImg ImgIdx] = TCCA_output(OutImg, ImgIdx, ...
            TCCANet.PatchSize(stage), TCCANet.NumFilters(stage), V{stage},NumView);  
    end
end

if IdtExt == 1 % enable feature extraction
    %display('PCANet training feature extraction...') 
    
    f1 = cell(NumImg,1); % compute the CCANet training feature one by one 
    f2 = cell(NumImg,1); % compute the CCANet training feature one by one
    f3 = cell(NumImg,1); % compute the CCANet training feature one by one
    
    for idx = 1:NumImg
        if 0==mod(idx,100); display(['Extracting TCCANet feasture of the ' num2str(idx) 'th training sample...']); end
        OutImgIndex = ImgIdx==idx; % select feature maps corresponding to image "idx" (outputs of the-last-but-one PCA filter bank) 
        
        outImg{1}=OutImg{1}(OutImgIndex);
        outImg{2}=OutImg{2}(OutImgIndex);
        outImg{3}=OutImg{3}(OutImgIndex);
        [OutImg_i ImgIdx_i] = TCCA_output(outImg, ones(sum(OutImgIndex),1),...
            TCCANet.PatchSize(end), TCCANet.NumFilters(end), V{end},NumView);  % compute the last PCA outputs of image "idx"
        [f1{idx} BlkIdx] = HashingHist(TCCANet,ImgIdx_i,OutImg_i{1}); % compute the feature of image "idx"
        [f2{idx} BlkIdx] = HashingHist(TCCANet,ImgIdx_i,OutImg_i{2}); % compute the feature of image "idx"
        [f3{idx} BlkIdx] = HashingHist(TCCANet,ImgIdx_i,OutImg_i{3}); % compute the feature of image "idx"
%        [f{idx} BlkIdx] = SphereSum(PCANet,ImgIdx_i,OutImg_i); % Testing!!
        OutImg{1}(OutImgIndex) = cell(sum(OutImgIndex),1); 
        OutImg{2}(OutImgIndex) = cell(sum(OutImgIndex),1); 
        OutImg{3}(OutImgIndex) = cell(sum(OutImgIndex),1);
    end
%     f=[f1{:}]+[f2{:}]+[f3{:}];             %FFS1
    f = sparse([f1{:};f2{:};f3{:}]);                  %FFS2
%     f = sparse([f1{:};f2{:}]); 
%     f = sparse([f3{:}]);
%     display(['t1= ' num2str(t1) 't2=' num2str(t2)]);
else  % disable feature extraction
    f = [];
    BlkIdx = [];
end

end

