clearvars -except looptime TrnSize TrainOrder result; close all;
addpath('./Liblinear');
NumView=3;
% TrnSize = 10000; 
TrnSize = 1000; 
ImgSize = 64; 
ImgFormat = 'gray'; %'color' or 'gray'
load('ETH-80.mat');
% clear RandViewG;
%get train data & valid data
View{1}=RandViewG;
clear RandViewG;
View{2}=RandViewR;
clear RandViewR;
View{3}=RandViewB;
clear RandViewB;
% Randomly choosing training&testing samples
randTrn = randperm(size(View{1,1},1));
RandTrn=randTrn(1:TrnSize);
label=View{1}(:,end);

TrnLabels = label(RandTrn);
label(RandTrn)=[];
TestLabels = label;
for i=1:3
    TrnData{i} = View{i}(RandTrn,1:end-1)';       % first view partition the data into training set and validation set
    View{i}(RandTrn,:)=[];
    TestData{i} = View{i}(:,1:end-1)';                       % first view partition the data into test set
end
% TrnData{2} = View{2}(RandTrn,1:end-1)';       % second view partition the data into training set and validation set
% View{2}(RandTrn,:)=[];
% TestData{2} = View{2}(:,1:end-1)';                       % second view partition the data into test set
% 
% TrnData{3} = View{3}(RandTrn,1:end-1)';       % third view partition the data into training set and validation set
% View{3}(RandTrn,:)=[];
% TestData{3} = View{3}(:,1:end-1)';                       % second view partition the data into test set
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
TCCANet.PatchSize = [7 7];
TCCANet.NumFilters = [8 8];
TCCANet.HistBlockSize = [7 7]; 
TCCANet.PchOverLapRatio = 0;
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
[ftrain V BlkIdx] = CCANet_train(TrnData_ImgCell,TCCANet,1,NumView); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
CCANet_TrnTime = toc;
clear TrnData_ImgCell;



%% CCANet Feature Extraction and Testing 
TestData_ImgCell{1} = mat2imgcell(TestData{1},ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
TestData_ImgCell{2} = mat2imgcell(TestData{2},ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
TestData_ImgCell{3} = mat2imgcell(TestData{3},ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
clear TestData; 


fprintf('\n ====== CCANet Testing ======= \n')

tic;
dist=zeros(nTestImg,TrnSize);
parfor idx = 1:1:nTestImg
    if 0==mod(idx,50); display(['Compute the distance of ' num2str(idx) 'th testing sample...']); end
    ftest = CCANet_FeaExt({TestData_ImgCell{1}{idx} TestData_ImgCell{2}{idx} TestData_ImgCell{3}{idx}},V,TCCANet,NumView); % extract a test feature using trained CCANet model 
     for numTrain=1:TrnSize
        dist(idx,numTrain)=norm(ftest-ftrain(:,numTrain));
     end
end
[D, order] = sort(dist,2);
K=1;
PredictLabel=zeros(nTestImg,K);
predictLabel=zeros(nTestImg,1);
parfor i=1:nTestImg
    for j=1:K
        PredictLabel(i,j)=TrnLabels(order(i,j));
    end
end
parfor i=1:nTestImg
    b=tabulate(PredictLabel(i,:));
    id=find(b(:,2)==max(b(:,2)));
    predictLabel(i)=id(1,1);
end
PCANet_TestTime = toc;
NumAccuracy=find((predictLabel-TestLabels)==0);
Accuracy=size(NumAccuracy,1)/nTestImg;
%compute the recognition accuracy for each class
display(['compute the recognition accuracy for each class...']); 
for idx = 1:1:nTestImg
    Accuracy_each_class(TestLabels(idx),2)=Accuracy_each_class(TestLabels(idx),2)+1;
    if predictLabel(idx) == TestLabels(idx)
        Accuracy_each_class(TestLabels(idx),1)=Accuracy_each_class(TestLabels(idx),1)+1;
    end
end
for i=1:ClassNum
    Accuracy_each_class(i,3)=Accuracy_each_class(i,1)/Accuracy_each_class(i,2);
end



