clear all; close all; clc; 
addpath('./Liblinear');
NumView=2;
% TrnSize = 10000; 
TrnSize = 500; 
ImgSize = 64; 
ImgFormat = 'gray'; %'color' or 'gray'
load('ETH-80.mat');
TrnLabels = RandViewR(1:TrnSize,end);
TestLabels = RandViewR(TrnSize+1:end,end);
clear RandViewG;
%get train data & valid data
View{1}=RandViewR;
clear RandViewR;
View{2}=RandViewB;
clear RandViewB;
TrnData{1} = View{1}(1:TrnSize,1:end-1)';       % first view partition the data into training set and validation set
TestData{1} = View{1}(TrnSize+1:end,1:end-1)';  % first view partition the data into test set
TrnData{2} = View{2}(1:TrnSize,1:end-1)';       % second view partition the data into training set and validation set
TestData{2} = View{2}(TrnSize+1:end,1:end-1)';  % second view partition the data into test set
clear View;

ClassNum = length(unique(TestLabels));       %ClassNum represents the total class number of test samples
Accuracy_each_class=zeros(ClassNum,3);       %the first column denotes the number of correct recognized and second column represents the total number of the row-th class 
                                             %the third column keep the
                                             %correct rate of the row-th
                                             %class

nTestImg = length(TestLabels);
%% CCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TPAMI submission
CCANet.NumStages = 2;
CCANet.PatchSize = [7 7];
CCANet.NumFilters = [8 8];
CCANet.HistBlockSize = [7 7]; 
CCANet.BlkOverLapRatio = 0.5;
CCANet.Pyramid = [];

fprintf('\n ====== CCANet Parameters ======= \n')
CCANet
%% CCANet Training with 10000 samples

fprintf('\n ====== CCANet Training ======= \n')
TrnData_ImgCell{1} = mat2imgcell(TrnData{1},ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
% clear TrnData_G;
TrnData_ImgCell{2} = mat2imgcell(TrnData{2},ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
clear TrnData;
% TrnData_ImgCell{1}=TrnData_ImgCell_C;
% TrnData_ImgCell{2}=TrnData_ImgCell_D;
tic;
[ftrain V BlkIdx] = CCANet_train(TrnData_ImgCell,CCANet,1,NumView); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
CCANet_TrnTime = toc;
clear TrnData_ImgCell;

fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
models = train(TrnLabels, ftrain', '-s 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;
clear ftrain; 


%% CCANet Feature Extraction and Testing 
TestData_ImgCell{1} = mat2imgcell(TestData{1},ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
% clear TestData_C; 
TestData_ImgCell{2} = mat2imgcell(TestData{2},ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
clear TestData; 
% TestData_ImgCell{1}=TestData_ImgCell_C;
% TestData_ImgCell{2}=TestData_ImgCell_D;
fprintf('\n ====== CCANet Testing ======= \n')

nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);

tic; 
for idx = 1:1:nTestImg
    
    testData_ImgCell{1}=TestData_ImgCell{1}{idx};
    testData_ImgCell{2}=TestData_ImgCell{2}{idx};
    ftest = CCANet_FeaExt(testData_ImgCell,V,CCANet,NumView); % extract a test feature using trained CCANet model 
    
    [xLabel_est, accuracy, decision_values] = predict(TestLabels(idx),...
        sparse(ftest'), models, '-q'); % label predictoin by libsvm
    
    Accuracy_each_class(TestLabels(idx),2)=Accuracy_each_class(TestLabels(idx),2)+1;
    if xLabel_est == TestLabels(idx)
        Accuracy_each_class(TestLabels(idx),1)=Accuracy_each_class(TestLabels(idx),1)+1;
        RecHistory(idx) = 1;
        nCorrRecog = nCorrRecog + 1;
    end
    
    if 0==mod(idx,nTestImg/100); 
        fprintf('Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \n',...
            [idx 100*nCorrRecog/idx toc/idx]); 
    end 
    
    TestData_ImgCell{1}{idx} = [];
    TestData_ImgCell{2}{idx} = [];
end
for i=1:ClassNum
    Accuracy_each_class(i,3)=Accuracy_each_class(i,1)/Accuracy_each_class(i,2);
end
Averaged_TimeperTest = toc/nTestImg;
Accuracy = nCorrRecog/nTestImg; 
ErRate = 1 - Accuracy;

%% Results display
fprintf('\n ===== Results of CCANet, followed by a linear SVM classifier =====');
fprintf('\n     CCANet training time: %.2f secs.', CCANet_TrnTime);
fprintf('\n     Linear SVM training time: %.2f secs.', LinearSVM_TrnTime);
fprintf('\n     Testing error rate: %.2f%%', 100*ErRate);
fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);



