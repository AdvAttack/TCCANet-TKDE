function [V] = HOSVD(M,NumFilters)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
m=cell(1,3);
u=cell(1,3);
V=cell(1,3);
for i=1:3
    m{i}=tenmat(M,i);
    [u{i},~,~]=svd(double(m{i}));
    V{i}=u{i}(:,1:NumFilters);
end

end
