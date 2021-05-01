function V = TCCA_FilterBank_broadcast(InImg, PatchSize, NumFilters, TCCANet, stage) 
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

eps=0.01;
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
for i = 1:ImgZ %1:ImgZ
    for j=1:3
        im{j}(:,(i-1)*NumCol+1:i*NumCol) = im2col_mean_removal(InImg{j}{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    end
end

%compute the variance matrix
for p=1:3
    Cpp{p}=(Cpp{p}+im{p}*im{p}')/N;
    Cpp_a{p}=Cpp{p}+I;
end

%compute the convariance tensor
if stage==1;
    a=im{1};
    b=im{2};
    c=im{3};
    clear im
    tic;
    parfor i=1:N
        C_123=C_123+reshape(kron(c(:,i),reshape(a(:,i)*b(:,i)',dim^2,1)),dim,dim,dim);
    end
    clear a b c;
    C_123=C_123/N;
    t_C123=toc;
else
    n=N/NumFilters;
    imSec=cell(8,3);
    for i=1:3
        for j=1:NumFilters
            imSec{j,i}=im{i}(:,(j-1)*n+1:j*n);
        end
    end
    clear im;
    tic;
    for j=1:NumFilters
        a=imSec{j,1};
        b=imSec{j,2};
        c=imSec{j,3};
        imSec{j,1}=[];
        imSec{j,2}=[];
        imSec{j,3}=[];
        parfor i=1:n
            C_123=C_123+reshape(kron(c(:,i),reshape(a(:,i)*b(:,i)',dim^2,1)),dim,dim,dim);
        end
        clear a b c;
    end
    C_123=C_123/N;
    t_C123=toc;
end
% M1=Cpp_a{1}^(-1/2)*tenmat(C_123,1)*(kron(Cpp_a{3}^(-1/2),Cpp_a{2}^(-1/2)))';
% M=reshape(double(M1),dim,dim,dim);
% M=ttm(ttm(ttm(tensor(C_123),Cpp_a{1}^(-1/2),1),Cpp_a{2}^(-1/2),2),Cpp_a{3}^(-1/2),3);
% M=ttm(ttm(ttm(tensor(C_123),Cpp_a{1}^(-1/2),1),Cpp_a{2}^(-1/2),2),Cpp_a{3}^(-1/2),3);
M=ttm(tensor(C_123),{Cpp_a{1}^(-1/2),Cpp_a{2}^(-1/2),Cpp_a{3}^(-1/2)});

%% ALS method
% v = ALS(M,NumFilters);
% M=tensor(M);
% P=cp_als_hosvd_initialized(M,NumFilters);
v = HOSVD(M,NumFilters);
% P=cp_als(M,NumFilters,'tol',1.0,'maxiters',1,'dimorder',[1 2 3],'init',v,'printitn',0);
P=cp_als(M,NumFilters,'init',v);
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

%% High Order Power Method(HOPM)
% M{1}=Cpp_a{1}^(-1/2)*tenmat(C_123,1)*kron(Cpp_a{3}^(-1/2),Cpp_a{2}^(-1/2));
% M{2}=Cpp_a{2}^(-1/2)*tenmat(C_123,2)*kron(Cpp_a{1}^(-1/2),Cpp_a{3}^(-1/2));
% M{3}=Cpp_a{3}^(-1/2)*tenmat(C_123,3)*kron(Cpp_a{2}^(-1/2),Cpp_a{1}^(-1/2));
% v = HOPM(M,NumFilters);
% V{1}=Cpp_a{1}^(-1/2)*v{1};
% V{2}=Cpp_a{2}^(-1/2)*v{2};
% V{3}=Cpp_a{3}^(-1/2)*v{3};


end


