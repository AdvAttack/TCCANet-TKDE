function V = TCCA_FilterBank_broadcast_loopC123(InImg, PatchSize, NumFilters, TCCANet, stage) 
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
addpath('tensor_toolbox');
[ImgX, ImgY]=size(InImg{1}{1});
NumCol=(ImgX-PatchSize+1)*(ImgY-PatchSize+1);
ImgZ = length(InImg{1});

eps=0.001;
dim=PatchSize^2;
N=NumCol*ImgZ;
I=eps*eye(dim,dim);
%% Learning TCCA filters (V)
%initialize
Cpp=cell(1,3);
Cpp_a=cell(1,3);
im=cell(1,3);
for p=1:3
    Cpp{p} = zeros(dim,dim);
    Cpp_a{p} = zeros(dim,dim);
    im{p}=zeros(dim,N);
end
C_123=reshape(zeros(PatchSize^6,1),dim,dim,dim);

% Construct sample matrices
tic;
for i = 1:ImgZ %1:ImgZ
    for j=1:3
        im{j}= im2col_mean_removal(InImg{j}{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
        Cpp{j}=Cpp{j}+im{j}*im{j}';
    end
    parfor k=1:NumCol
        C_123=C_123+reshape(kron(im{3}(:,k),reshape(im{1}(:,k)*im{2}(:,k)',dim^2,1)),dim,dim,dim);
    end
end
C_123=C_123/N;
t_C123=toc;
% compute the variance matrix
for p=1:3
    Cpp{p}=Cpp{p}/N;
    Cpp_a{p}=Cpp{p}+I;
end

M=ttm(tensor(C_123),{Cpp_a{1}^(-1/2),Cpp_a{2}^(-1/2),Cpp_a{3}^(-1/2)});

%% ALS method
% P=cp_als_hosvd_initialized(M,NumFilters);
P=cp_als(M,NumFilters);
for i=1:3
    v{i}=P.U{i};
end
% [U,iter,REC] = ncp(tensor(M),NumFilters);
%% HOSVD method
% v = HOSVD(M,NumFilters);
V=cell(1,3);
for p=1:3
    V{p}=Cpp_a{p}^(-1/2)*v{p};
end


end