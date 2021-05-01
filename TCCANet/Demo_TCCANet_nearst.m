clearvars -except looptime TrnSize TrainOrder result; close all;
addpath('./Liblinear');
addpath('./tensor_toolbox');
NumView=3;
% TrnSize = 10000; 
TrnSize = 1000; 
ImgSize = 64; 
ImgFormat = 'gray'; %'color' or 'gray'
load('RSSCN7_LBP_WT_edge_original.mat')
clear view1_LBP;
%get train data & valid data
View{1}=view2_WT;
clear view2_WT;
View{2}=view3_edge;
clear view3_edge;
View{3}=view4_original;
clear view4_original;
% Randomly choosing training&testing samples
randTrn = randperm(size(View{1,1},1));
RandTrn=randTrn(1:TrnSize);
label=View{1}(:,end);

TrnLabels = label(RandTrn);
label(RandTrn)=[];
TestLabels = label;
TrnData=cell(1,3);
TestData=cell(1,3);
for i=1:3
    TrnData{i} = View{i}(RandTrn,1:end-1)';       % first view partition the data into training set and validation set
    View{i}(RandTrn,:)=[];
    TestData{i} = View{i}(:,1:end-1)';                       % first view partition the data into test set
end
clear View;

ClassNum = length(unique(TestLabels));       %ClassNum represents the total class number of test samples
Accuracy_each_class=zeros(ClassNum,3);       %the first column denotes the number of correct recognized and second column represents the total number of the row-th class 
                                             %the third column keep the
                                             %correct rate of the row-th
                                             %class

nTestImg = length(TestLabels);
%% CCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TPAMI submission
TCCANet.NumStages = 2;
TCCANet.PatchSize = [5 5];
TCCANet.NumFilters = [8 8];
TCCANet.HistBlockSize = [31 31]; 
TCCANet.BlkOverLapRatio = 0.5;
TCCANet.Pyramid = [];

fprintf('\n ====== TCCANet Parameters ======= \n')
TCCANet
%% CCANet Training with 10000 samples

fprintf('\n ====== TCCANet Training ======= \n')
TrnData_ImgCell{1} = mat2imgcell(TrnData{1},ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
TrnData_ImgCell{2} = mat2imgcell(TrnData{2},ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
TrnData_ImgCell{3} = mat2imgcell(TrnData{3},ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
clear TrnData;

tic;
[ftrain V BlkIdx] = TCCANet_train(TrnData_ImgCell,TCCANet,1,NumView); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
TCCANet_TrnTime = toc;
clear TrnData_ImgCell;


fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
models = train(TrnLabels, ftrain', '-s 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;
clear ftrain; 


%% CCANet Feature Extraction and Testing 
TestData_ImgCell{1} = mat2imgcell(TestData{1},ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
TestData_ImgCell{2} = mat2imgcell(TestData{2},ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
TestData_ImgCell{3} = mat2imgcell(TestData{3},ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
clear TestData; 
fprintf('\n ====== TCCANet Testing ======= \n')

nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);

tic; 
for idx = 1:1:nTestImg
    
    testData_ImgCell{1}=TestData_ImgCell{1}{idx};
    testData_ImgCell{2}=TestData_ImgCell{2}{idx};
    testData_ImgCell{3}=TestData_ImgCell{3}{idx};
    ftest = TCCANet_FeaExt(testData_ImgCell,V,TCCANet,NumView); % extract a test feature using trained CCANet model 
    
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
    TestData_ImgCell{3}{idx} = [];
end
for i=1:ClassNum
    Accuracy_each_class(i,3)=Accuracy_each_class(i,1)/Accuracy_each_class(i,2);
end
Averaged_TimeperTest = toc/nTestImg;
Accuracy = nCorrRecog/nTestImg; 
ErRate = 1 - Accuracy;

%% Results display
fprintf('\n ===== Results of TCCANet, followed by a linear SVM classifier =====');
fprintf('\n     TCCANet training time: %.2f secs.', TCCANet_TrnTime);
fprintf('\n     Linear SVM training time: %.2f secs.', LinearSVM_TrnTime);
fprintf('\n     Testing error rate: %.2f%%', 100*ErRate);
fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);















