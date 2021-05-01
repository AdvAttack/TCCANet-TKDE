TrnSize=1000;
result=zeros(8,10);
TrainOrder=zeros(TrnSize,10);
for looptime=1:10
    fprintf(['\n ======LOOP FOR THE ' num2str(looptime) 'TH TIME======= \n'])
    Demo_TCCANet_nearst;
    result(1,looptime)=Accuracy;
    result(2:8,looptime)=Accuracy_each_class(:,3);
    TrainOrder(:,looptime)=RandTrn';
    display(['Recognition Accuracy= ' num2str(Accuracy) 'For THE ' num2str(looptime) 'TH TIME...']); 
end
Result.result=result;
Result.TrainOrder=TrainOrder;
% save(strcat('CollegeSever_AdjustParam_PatchSize11*11_HistBlockSize3*3.mat'),'Result');
save(strcat('CollegeSever_TCCANet_',num2str(TrnSize),'Trn_cp_als.mat'),'Result');