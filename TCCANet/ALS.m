function [U] = ALS(M,NumFilters)
%UNTITLED Summary of this function goes here
% =======INPUT=============
%        M: The tensor 
%        NumFilters: Number of Filters
%        n: The loop time of ALS algorithm
% =======OUTPUT============
%        U:The best rank-1 approximation of M
%% Random initialization
% for i=1:3
%     U{i}=rand(size(M,i),NumFilters);
% end
%% Initialize the left singular matrix
for i=1:3
    m{i}=tenmat(M,i);
    [u{i} s v]=svd(double(m{i}));
    U{i}=u{i}(:,1:NumFilters);
end
clear s v
% for i=1:5
%     U{1}= (pinv(KhatriProduct( U{2},U{3} ))*double(tenmat(M,1)'))';
%     for j=1:NumFilters
%         lamda1(j)=norm(U{1}(:,j));
%         U{1}(:,j)=U{1}(:,j)/lamda1(j);
%     end
%     U{2}= (pinv(KhatriProduct( U{3},U{1} ))*double(tenmat(M,2)'))';
%     for k=1:NumFilters
%         lamda2(k)=norm(U{2}(:,k));
%         U{2}(:,k)=U{2}(:,k)/lamda2(k);
%     end
%     U{3}= (pinv(KhatriProduct( U{1},U{2} ))*double(tenmat(M,3)'))';
%     for l=1:NumFilters
%         lamda3(l)=norm(U{3}(:,l));
%         U{3}(:,l)=U{3}(:,l)/lamda3(l);
%     end
% end

end

