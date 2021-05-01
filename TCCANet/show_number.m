% show the number of i-th train sample
i=3;
vec1=Daub_mnist_test(i,1:end-1);
I=reshape(vec1,28,28)';%map the vector to the corresponding matrix
I2=255*I;              %normalization
imshow(I2);