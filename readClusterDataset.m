% clustering dataset reading
% dataset_name: NUS_lite, MSRCV1

function [data, label] = readClusterDataset(dataset_name, varargin)

switch dataset_name
    case 'NUS_lite'
        label_path = '..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\ground truth\'; 
        train_path = {'..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Train_CH.txt',...
                    '..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Train_EDH.txt',...
                    '..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Train_WT.txt'};
        test_path = {'..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Test_CH.txt',...
                     '..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\TEST_EDH.txt',...
                     '..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Test_WT.txt'};

        data_path = {train_path,test_path};
        k = numel(train_path);

        Ytrn = double(ReadLabel(label_path, 'Train'));

        Ytst = double(ReadLabel(label_path, 'Test'));

        nconcept = size(Ytrn,2);

        for i = 1:k
            X0{i} = [ReadData(train_path{i}),ReadData(test_path{i})];
            kf(i) = size(X0{i},1);
        end
    case 'MSRCV1'
        load('..\Project_X\MSRCV1.mat');
      
        Ytrn = Y_train;
        Ytst = Y_test;
        k = numel(X_train);
        for i = 1:k
            X0{i} = [X_train{i},X_test{i}];
        end
        
    case 'AWA'   
        [X_train, X_test, Ytrn, Ytst] = ReadDataSetAWA;
         k = numel(X_train);
         for i = 1:k
            X0{i} = [X_train{i},X_test{i}];
        end
end

%% ------------normalization-----------
%k = numel(X0);
c = size(Ytrn,2);
ntrn = size(Ytrn,1);
ntst = size(Ytst,1);
n = ntrn+ntst;

for i = 1:k
    X{i} = (X0{i}-repmat(min(X0{i},[],2),1,n))./...
        repmat(max(X0{i},[],2) - min(X0{i},[],2),1,n);
    Xtrn{i} = X{i}(:,1:ntrn);
    Xtst{i} = X{i}(:,ntrn+1:n);
end

data = X0;
label = [Ytrn; Ytst];