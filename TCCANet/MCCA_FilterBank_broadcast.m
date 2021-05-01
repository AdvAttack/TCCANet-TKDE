function V = MCCA_FilterBank_broadcast(InImg, PatchSize, NumFilters, TCCANet, stage) 
% =======INPUT=============
% InImg            Input images (cell structure)  
% PatchSize        the patch size, asumed to an odd number.
% NumFilters       the number of PCA filters in the bank.
% =======OUTPUT============
% V                PCA filter banks, arranged in column-by-column manner
% =========================

% addpath('./Utils')

% to efficiently cope with the large training samples, if the number of training we randomly subsample 10000 the
% training set to learn PCA filter banks
% addpath('/home/xhyang/文档/TCCA/tensor_toolbox');
[ImgX, ImgY]=size(InImg{1}{1});
NumCol=(ImgX-PatchSize+1)*(ImgY-PatchSize+1);
ImgZ = length(InImg{1});

eps=0.001;
dim=PatchSize^2;
dim_SL=dim*3;
N=NumCol*ImgZ;
I=eps*eye(dim_SL,dim_SL);
%% Learning TCCA filters (V)
%initialize
im=cell(1,3);
for p=1:3
    im{p}=zeros(dim,N);
end

% Construct sample matrices
for i = 1:ImgZ %1:ImgZ
    for j=1:3
        im{j}(:,(i-1)*NumCol+1:i*NumCol) = im2col_mean_removal(InImg{j}{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    end
end

%================================计算Sii==================================%
for i=1:3
    Sii{i}=(im{i}*im{i}')/(N);  %计算local within-view covariance matrix：Sii
end
%================================计算Sij==================================%
for i=1:3
    for j=1:3
        Sij{i,j}=(im{i}*im{j}')/(N);%计算Sij(150,150)
    end
end

%=============================calculate SdL&SL============================%
SdL=Sii{1};                         %initialize Sdl
for i=2:3
    SdL=blkdiag(SdL,Sii{i});        %give SdL assignment
end
SL=cell2mat(Sij);                   %give SL assignment

%==========================SdL is singular matrix?========================%
Rank=rank(SdL);                     %Rank:the rank of SdL
if Rank~=dim_SL
    SdL=SdL+I;                      %if SdL is a singular matrix, then process SdL like this
    [V,D]=eig(SL,SdL);              %D:eigenvalues  V:columns are the corresponding eigenvectors
else
    [V,D]=eig(SL,SdL);              %D:eigenvalues  V:columns are the corresponding eigenvectors
end
% 对特征值排序
[sort_D,order]=sort(diag(D),'descend');
v=zeros(dim_SL,dim_SL);
for i=1:dim_SL
    v(:,i)=V(:,order(i));
end
sort_V=v;
clear V;
for i=1:3
    V{i}=sort_V((i-1)*dim+1:i*dim,1:NumFilters);
end




end



