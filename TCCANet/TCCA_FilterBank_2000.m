function V = TCCA_FilterBank_2000(InImg, PatchSize, NumFilters, TCCANet, stage) 
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
addpath('/home/xhyang/文档/TCCA/tensor_toolbox');
[ImgX, ImgY]=size(InImg{1}{1});
NumCol=(ImgX-PatchSize+1)*(ImgY-PatchSize+1);
ImgZ = length(InImg{1});

eps=0.001;
dim=PatchSize^2;
N=NumCol*ImgZ;
I=eps*eye(dim,dim);
%% Learning TCCA filters (V)
%initialize
for p=1:3
    Cpp{p} = zeros(dim,dim);
    Cpp_a{p} = zeros(dim,dim);
    im{p}=zeros(dim,N);
end
C_123=reshape(zeros(PatchSize^6,1),dim,dim,dim);

for i = 1:ImgZ %1:ImgZ
    for j=1:3
        im{j}(:,(i-1)*NumCol+1:(i*NumCol)) = im2col_mean_removal(InImg{j}{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    end
end

%compute the variance matrix
for p=1:3
    Cpp{p}=(Cpp{p}+im{p}*im{p}')/N;
    Cpp_a{p}=Cpp{p}+I;
end

%compute the convariance tensor
if stage==1;
    n=N/2;
    for i=1:3
        im1{i}=im{1}(:,1:n);
        im2{i}=im{i}(:,n+1:2*n);
    end
    clear im;
    tic;
    parfor i=1:n
        C_123=C_123+reshape(kron(im1{3}(:,i),reshape(im1{1}(:,i)*im1{2}(:,i)',2401,1)),49,49,49);
    end
    clear im1;
    parfor i=1:n
        C_123=C_123+reshape(kron(im2{3}(:,i),reshape(im2{1}(:,i)*im2{2}(:,i)',2401,1)),49,49,49);
    end
    clear im2;
    C_123=C_123/N;
    t_C123=toc;
else
    n=N/(NumFilters*2);
    for i=1:3
        im1{i}=im{i}(:,1:n);      
        im2{i}=im{i}(:,n+1:2*n);
        im3{i}=im{i}(:,2*n+1:3*n);
        im4{i}=im{i}(:,3*n+1:4*n);
        im5{i}=im{i}(:,4*n+1:5*n);
        im6{i}=im{i}(:,5*n+1:6*n);
        im7{i}=im{i}(:,6*n+1:7*n);
        im8{i}=im{i}(:,7*n+1:8*n);
        im9{i}=im{i}(:,8*n+1:9*n);      
        im10{i}=im{i}(:,9*n+1:10*n);
        im11{i}=im{i}(:,10*n+1:11*n);
        im12{i}=im{i}(:,11*n+1:12*n);
        im13{i}=im{i}(:,12*n+1:13*n);
        im14{i}=im{i}(:,13*n+1:14*n);
        im15{i}=im{i}(:,14*n+1:15*n);
        im16{i}=im{i}(:,15*n+1:16*n);
    end
    clear im;
    tic;
    parfor i=1:n
        C_123=C_123+reshape(kron(im1{3}(:,i),reshape(im1{1}(:,i)*im1{2}(:,i)',2401,1)),49,49,49);
    end
    clear im1;
    parfor i=1:n
        C_123=C_123+reshape(kron(im2{3}(:,i),reshape(im2{1}(:,i)*im2{2}(:,i)',2401,1)),49,49,49);
    end
    clear im2;
    parfor i=1:n
        C_123=C_123+reshape(kron(im3{3}(:,i),reshape(im3{1}(:,i)*im3{2}(:,i)',2401,1)),49,49,49);
    end
    clear im3;
    parfor i=1:n
        C_123=C_123+reshape(kron(im4{3}(:,i),reshape(im4{1}(:,i)*im4{2}(:,i)',2401,1)),49,49,49);
    end
    clear im4;
    parfor i=1:n
        C_123=C_123+reshape(kron(im5{3}(:,i),reshape(im5{1}(:,i)*im5{2}(:,i)',2401,1)),49,49,49);
    end
    clear im5;
    parfor i=1:n
        C_123=C_123+reshape(kron(im6{3}(:,i),reshape(im6{1}(:,i)*im6{2}(:,i)',2401,1)),49,49,49);
    end
    clear im6;
    parfor i=1:n
        C_123=C_123+reshape(kron(im7{3}(:,i),reshape(im7{1}(:,i)*im7{2}(:,i)',2401,1)),49,49,49);
    end
    clear im7;
    parfor i=1:n
        C_123=C_123+reshape(kron(im8{3}(:,i),reshape(im8{1}(:,i)*im8{2}(:,i)',2401,1)),49,49,49);
    end
    clear im8;
    parfor i=1:n
        C_123=C_123+reshape(kron(im9{3}(:,i),reshape(im9{1}(:,i)*im9{2}(:,i)',2401,1)),49,49,49);
    end
    clear im9;
    parfor i=1:n
        C_123=C_123+reshape(kron(im10{3}(:,i),reshape(im10{1}(:,i)*im10{2}(:,i)',2401,1)),49,49,49);
    end
    clear im10;
    parfor i=1:n
        C_123=C_123+reshape(kron(im11{3}(:,i),reshape(im11{1}(:,i)*im11{2}(:,i)',2401,1)),49,49,49);
    end
    clear im11;
    parfor i=1:n
        C_123=C_123+reshape(kron(im12{3}(:,i),reshape(im12{1}(:,i)*im12{2}(:,i)',2401,1)),49,49,49);
    end
    clear im12;
    parfor i=1:n
        C_123=C_123+reshape(kron(im13{3}(:,i),reshape(im13{1}(:,i)*im13{2}(:,i)',2401,1)),49,49,49);
    end
    clear im13;
    parfor i=1:n
        C_123=C_123+reshape(kron(im14{3}(:,i),reshape(im14{1}(:,i)*im14{2}(:,i)',2401,1)),49,49,49);
    end
    clear im14;
    parfor i=1:n
        C_123=C_123+reshape(kron(im15{3}(:,i),reshape(im15{1}(:,i)*im15{2}(:,i)',2401,1)),49,49,49);
    end
    clear im15;
    parfor i=1:n
        C_123=C_123+reshape(kron(im16{3}(:,i),reshape(im16{1}(:,i)*im16{2}(:,i)',2401,1)),49,49,49);
    end
    clear im16;
    C_123=C_123/N;
    t_C123=toc;
end
M1=Cpp_a{1}^(-1/2)*tenmat(C_123,1)*(kron(Cpp_a{2}^(-1/2),Cpp_a{3}^(-1/2)))';
M=reshape(double(M1),dim,dim,dim);
% M=ttm(ttm(ttm(tensor(C_123),Cpp_a{1}^(-1/2),1),Cpp_a{2}^(-1/2),2),Cpp_a{3}^(-1/2),3);

%% ALS method
v = ALS(M,NumFilters);
% M=tensor(M);
% P=cp_als_hosvd_initialized(M,NumFilters);
% for i=1:3
%     v{i}=P.U{i};
% end
% [U,iter,REC] = ncp(tensor(M),NumFilters);
%% HOSVD method
% v = HOSVD(M,NumFilters);
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


