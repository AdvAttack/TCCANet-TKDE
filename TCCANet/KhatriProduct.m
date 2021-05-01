function [ c ] = KhatriProduct( a,b )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
for i=1:size(a,2)
    c(:,i)=kron(a(:,i),b(:,i));
end

end

