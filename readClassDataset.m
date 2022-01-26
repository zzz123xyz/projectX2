% create, normalize and read dataset for classification task
% input:
%   dataset_name : dataset name (choose from existing datasets)
%   varargin : percentage of the testing samples (optional)
% output:
%   feat_trn: Trainning features (ndim*nsamples_trn)
%   feat_tst: Testing features (ndim*nsamples_tst)
%   label_trn: Trainning labels (1*nsamples_trn)
%   label_tst: Testing labels (1*nsamples_tst)
%   trainIndicator: Trainning samples indicators (1*nsamples binary)
%   testIndicator: Testing samples indicators (1*nsamples binary)

function [feat_trn, feat_tst, label_trn, label_tst, trainIndicator, testIndicator] = readClassDataset(dataset_name_full, varargin)

tmp = strfind(dataset_name_full,'_');
if isempty(tmp)
    dataset_name = dataset_name_full;
else
    dataset_name = dataset_name_full(1:tmp(1)-1);
end


if numel(varargin) ~= 0
    ID = varargin{1};
else
    ID = 1; % default
end

switch dataset_name
    %     case 'NUSlite'
    %         label_path = '..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\ground truth\';
    %         train_path = {'..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Train_CH.txt',...
    %                     '..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Train_EDH.txt',...
    %                     '..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Train_WT.txt'};
    %         test_path = {'..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Test_CH.txt',...
    %                      '..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\TEST_EDH.txt',...
    %                      '..\Project_X\NUS_WIDE_lite\NUS-WIDE-SCENE\low level features\Test_WT.txt'};
    %
    %         data_path = {train_path,test_path};
    %         k = numel(train_path);
    %
    %         Ytrn = double(ReadLabel(label_path, 'Train'));
    %
    %         Ytst = double(ReadLabel(label_path, 'Test'));
    %
    %         nconcept = size(Ytrn,2);
    %
    %         for i = 1:k
    %             X0{i} = [ReadData(train_path{i}),ReadData(test_path{i})];
    %             kf(i) = size(X0{i},1);
    %         end
    %     case 'NUSWIDEOBJ'
    %         load('../dataset_Large-Scale/NUSWIDEOBJ.mat');
    %         X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;
    %         k = numel(X0);
    
    case 'MSRCV1'
        load('..\Project_X\MSRCV1.mat');
        
        [~, label_trn] = find(Y_train);
        [~, label_tst] = find(Y_test);
        
        trainIndicator = logical([ones(size(Y_train,1),1);zeros(size(Y_test,1),1)]);
        testIndicator = logical([zeros(size(Y_train,1),1);ones(size(Y_test,1),1)]);
        k = numel(X_train);
        for i = 1:k
            X0{i} = [X_train{i},X_test{i}];
        end
        
        X0 = cellfun(@transpose, X0, 'UniformOutput', false); clear X;
        X0 = cell2mat(X0);
        X0 = X0';
        
        X0 = DataNormalization(X0, 1);
        feat_trn = X0(:,trainIndicator);
        feat_tst = X0(:,testIndicator);
        
    case 'AWA'
        [X_train, X_test, Ytrn, Ytst] = ReadDataSetAWA;
        k = numel(X_train);
        for i = 1:k
            X0{i} = [X_train{i},X_test{i}];
        end
        
    case 'ApAy'
        if strcmp(dataset_name_full, 'ApAy')  % for loading ApAy original reduced features 
            [~, ~, Y, ~] = ReadDataSetApAy;
            comp_data_file = ['../computed_data/feat_reduce_',dataset_name,'.mat'];
            load(comp_data_file); % load feat_trn_red, feat_tst_red
            X0 = DataNormalization(feat_trn_red, 1);
            
            file_path = ['..\computed_data\ApAy_MDR_R01R01R005'];
            load(file_path); % load Data, testIndicator, trainIndicator;
            
            feat_trn = X0( : , trainIndicator);
            feat_tst = X0( : , testIndicator);
            
            label_trn = Y(trainIndicator);
            label_tst = Y(testIndicator);
            
        elseif strcmp(dataset_name_full, 'ApAy_5000')...
                ||strcmp(dataset_name_full, 'ApAy_5000_neMVSC')...
                ||strcmp(dataset_name_full, 'ApAy_5000_neCLR') % for loading ApAy original reduced features     
            [~, ~, Ytrn, Ytst] = ReadDataSetApAy;
            dataset_name = 'ApAy';
            comp_data_file = ['../computed_data/feat_reduce_',dataset_name,'.mat'];
            load(comp_data_file); % load feat_trn_red, feat_tst_red
            X0 = DataNormalization(feat_trn_red, 1);
            
            file_path = ['..\computed_data\ApAy_MDR_R01R01R005'];
            load(file_path); % load Data, testIndicator, trainIndicator;
            
            feat_trn = X0( : , trainIndicator);
            feat_tst = feat_tst_red;
            
            label_trn = Ytrn(trainIndicator);
            label_tst = Ytst;
            
        else  % for loading ApAy_MDR_R01R01R005 features
            [~, ~, Y, ~] = ReadDataSetApAy;
            
            file_path = ['..\computed_data\',dataset_name_full];
            load(file_path); % load Data, testIndicator, trainIndicator;
            
            % X0 = cellfun(@transpose, Data, 'UniformOutput', false); clear Data;
            X0 = cell2mat(Data');
            X0 = DataNormalization(X0, 1);
            
            feat_trn = X0( : , trainIndicator);
            feat_tst = X0( : , testIndicator);
            
            label_trn = Y(trainIndicator);
            label_tst = Y(testIndicator);
            
        end
        
    case 'Cal7'
        load('../dataset_Large-Scale/Caltech101-7.mat');
        load(['datasets_split/',ls(['datasets_split/',dataset_name,'_Split_ID',num2str(ID,'%03i'),'*'])])
        
        X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;
        X0 = cell2mat(X0');
        X0 = DataNormalization(X0, 1);
        
        feat_trn = X0(:, trainIndicator);
        feat_tst = X0(:, testIndicator);
        
        label_trn = Y(trainIndicator);
        label_tst = Y(testIndicator);
        
    case 'Cal20'
        load('../dataset_Large-Scale/Caltech101-20.mat');
        load(['datasets_split/',ls(['datasets_split/',dataset_name,'_Split_ID',num2str(ID,'%03i'),'*'])])
        
        X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;
        X0 = cell2mat(X0');
        X0 = DataNormalization(X0, 1);
        
        feat_trn = X0(:, trainIndicator);
        feat_tst = X0(:, testIndicator);
        
        label_trn = Y(trainIndicator);
        label_tst = Y(testIndicator);
        
    case 'HW'
        load('../dataset_Large-Scale/handwritten.mat');
        load(['datasets_split/',ls(['datasets_split/',dataset_name,'_Split_ID',num2str(ID,'%03i'),'*'])])
        
        X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;
        X0 = cell2mat(X0');
        X0 = DataNormalization(X0, 1);
        
        Y = Y + 1;
        
        feat_trn = X0(:, trainIndicator);
        feat_tst = X0(:, testIndicator);
        
        label_trn = Y(trainIndicator);
        label_tst = Y(testIndicator);
        
    case 'AWA4000'
        load('V:\lance\Animals_with_Attributes\AWA4000\AWA4000.mat');
        X0 = X; clear X;
        k = numel(X0);
        
    case 'USAA'
        file_path = '..\USAA.mat';
        load(file_path);
        
        label_trn = train_video_label;
        label_tst = test_video_label;
        
        if strcmp(dataset_name_full, 'USAA')
            % uncomplete
            
            Xtrn = Xtrain'; clear Xtrain
            Xtst = Xtest'; clear Xtest
            X = [Xtrn,Xtst];
            X0{1} = X(1:5000,:);
            X0{2} = X(5001:10000,:);
            X0{3} = X(10001:14000,:);
            k = numel(X0);
            clear X;
            
        else
            file_path = ['..\computed_data\',dataset_name_full];
            load(file_path);
            
            % X0 = cellfun(@transpose, Data, 'UniformOutput', false); clear Data;
            X0 = cell2mat(Data');
            X0 = DataNormalization(X0, 1);
            
            ntrn = 735;
            ntst = 731;
            trainIndicator = [1:ntrn];
            testIndicator = [ntrn + 1: (ntrn + ntst)];
            
            feat_trn = X0( : , trainIndicator);
            feat_tst = X0( : , testIndicator);
            
        end
        
end


