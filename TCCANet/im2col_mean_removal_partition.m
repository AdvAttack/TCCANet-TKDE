function [ im ] = im2col_mean_removal_partition(InImg, PatchSize, stride) 
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[ImgX,ImgY]=size(InImg);
%InImg=gpuArray(InImg);
RowBlock=round((ImgX-PatchSize)/stride+1);
ColBlock=round((ImgY-PatchSize)/stride+1);
im=zeros(PatchSize^2,RowBlock*ColBlock);
for i=1:RowBlock
    for j=1:ColBlock
        Block=InImg((i-1)*stride+1:(i-1)*stride+PatchSize,(j-1)*stride+1:(j-1)*stride+PatchSize);
        im(:,(i-1)*RowBlock+j)=reshape(Block,PatchSize^2,1);
    end
end
DimMean=repmat(mean(im,2),1,RowBlock*ColBlock);
im=im-DimMean;

end

