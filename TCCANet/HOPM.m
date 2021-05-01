function [V] = HOPM(M,NumFilters)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
for i=1:3
    M{i}=double(M{i});
end
%initialize
[u S V]=svd(M{2});
U0{2}=u(:,1);
[u S V]=svd(M{3});
U0{3}=u(:,1);
clear S V;
%the first loop
Ua{1}{1}=M{1}*(kron(U0{2}',U0{3}'))';
lamda1(1)=norm(Ua{1}{1});
U{1}{1}=Ua{1}{1}/lamda1(1);

Ua{2}{1}=M{2}*(kron(U0{3}',U{1}{1}'))';
lamda2(1)=norm(Ua{2}{1});
U{2}{1}=Ua{2}{1}/lamda2(1);

Ua{3}{1}=M{3}*(kron(U{1}{1}',U{2}{1}'))';
lamda3(1)=norm(Ua{3}{1});
U{3}{1}=Ua{3}{1}/lamda3(1);

%for the 2nd loop
Ua{1}{2}=M{1}*(kron(U{2}{1}',U{3}{1}'))';
lamda1(2)=norm(Ua{1}{2});
U{1}{2}=Ua{1}{2}/lamda1(2);
    
Ua{2}{2}=M{2}*(kron(U{3}{1}',U{1}{2}'))';
lamda2(2)=norm(Ua{2}{2});
U{2}{2}=Ua{2}{2}/lamda2(2);
    
Ua{3}{2}=M{3}*(kron(U{1}{2}',U{2}{2}'))';
lamda3(2)=norm(Ua{3}{2});
U{3}{2}=Ua{3}{2}/lamda3(2);


%i=2;
%epsilon=0.001;
%while (lamda1(i)-lamda1(i-1)>epsilon)||(lamda2(i)-lamda2(i-1)>epsilon)||(lamda3(i)-lamda3(i-1)>epsilon)
%    i=i+1;
%    Ua{1}{i}=M{1}*(kron(U{2}{i-1}',U{3}{i-1}'))';
 %   lamda1(i)=norm(Ua{1}{i});
%    U{1}{i}=Ua{1}{i}/lamda1(i);
    
%    Ua{2}{i}=M{2}*(kron(U{3}{i-1}',U{1}{i}'))';
%    lamda2(i)=norm(Ua{2}{i});
%    U{2}{i}=Ua{2}{i}/lamda2(i);
    
%    Ua{3}{i}=M{3}*(kron(U{1}{i}',U{2}{i}'))';
%    lamda3(i)=norm(Ua{3}{i});
 %   U{3}{i}=Ua{3}{i}/lamda3(i);
%end

for i=3:NumFilters
    Ua{1}{i}=M{1}*(kron(U{2}{i-1}',U{3}{i-1}'))';
    lamda1(i)=norm(Ua{1}{i});
    U{1}{i}=Ua{1}{i}/lamda1(i);
    
    Ua{2}{i}=M{2}*(kron(U{3}{i-1}',U{1}{i}'))';
    lamda2(i)=norm(Ua{2}{i});
    U{2}{i}=Ua{2}{i}/lamda2(i);
    
    Ua{3}{i}=M{3}*(kron(U{1}{i}',U{2}{i}'))';
    lamda3(i)=norm(Ua{3}{i});
    U{3}{i}=Ua{3}{i}/lamda3(i);
end

for i=1:3
    V{i}=cell2mat(U{i});
end

end


