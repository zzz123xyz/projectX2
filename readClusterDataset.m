%% clustering dataset reading
% --- details --- (option)

% --- version ---- (option)

% --- Input ---

% --- output ----
% data: 
%   a cell with nfeat elements which are in form of (ndim * nsample) 
% label:
%   label vector of the data (1 * nsample)
% --- ref ---

% --- note ---(option)
% 1. see also readClassDataset.m
% 2. line  if strcmp(dataset_name_full, 'ApAy') %% Do I need to use the first part of the if section ???
%

% by Lance Liu 

function [data, label] = readClusterDataset(dataset_name_full, varargin)


tmp = strfind(dataset_name_full,'_');
if isempty(tmp)
    dataset_name = dataset_name_full;
else
    dataset_name = dataset_name_full(1:tmp(1)-1);
end

if ismember(dataset_name, {'NUS_lite', 'MSRCV1', 'AWA', 'ApAy', 'USAA', 'USPS'})
    DatasetType = 1;
elseif ismember(dataset_name, {'Cal7', 'Cal20', 'HW' ,'NUSWIDEOBJ', 'AWA4000','ApAy_MDR','AWA_MDR'})
    DatasetType = 2;
end

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
    case 'NUSWIDEOBJ'    
        load('../dataset_Large-Scale/NUSWIDEOBJ.mat');
        X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;       
        k = numel(X0);
        
    case 'MSRCV1'
        load('..\Project_X\MSRCV1.mat');

        Ytrn = Y_train;
        Ytst = Y_test;
        k = numel(X_train);
        for i = 1:k
            X0{i} = [X_train{i},X_test{i}];
        end
        
    case 'ASUN' % uncomplete
        comp_label_file = ['../computed_data/label_',dataset_name,'.mat'];
        load(comp_label_file);
        Ytrn = img_classes_num_trn;
        Ytst = img_classes_num_tst;        
        
        
        
        comp_data_file = ['../computed_data/feat_reduce_',dataset_name,'.mat'];
        load(comp_data_file);
        
        X_train = {feat_trn_red'};
        X_test = {feat_tst_red'};
         k = numel(X_train);
         for i = 1:k
            X0{i} = [X_train{i};X_test{i}]';
         end

    case 'AWA'
        [X_train, X_test, Ytrn, Ytst] = ReadDataSetAWA;
         k = numel(X_train);
         for i = 1:k
            X0{i} = [X_train{i},X_test{i}];
         end

    case 'ApAy'
        
        if strcmp(dataset_name_full, 'ApAy')||~isempty(strfind(dataset_name_full,'ApAy_cnn'))
            %% Do I need to use the first part of the if section ???
            [~, ~, Ytrn, Ytst] = ReadDataSetApAy;
            comp_data_file = ['../computed_data/feat_reduce_',dataset_name,'.mat'];
            load(comp_data_file); %load feat_trn_red, feat_tst_red
            X_train = {feat_trn_red'};
            X_test = {feat_tst_red'};
            k = numel(X_train);
            for i = 1:k
                X0{i} = [X_train{i};X_test{i}]';
            end
            
        elseif strcmp(dataset_name_full, 'ApAy_4_trn')    
            [~, ~, Y, ~] = ReadDataSetApAy;
            filepath = ['../computed_data/',dataset_name_full];
            load(filepath);  %load Data
            X0 = Data; clear Data
            k = numel(X0);
            DatasetType = 2;
            
        else
            [~, ~, Y, ~] = ReadDataSetApAy; % only consider training samples
            filepath = ['../computed_data/',dataset_name_full];
            load(filepath); %load Data, testIndicator, trainIndicator
            X0 = Data; clear Data
            k = numel(X0);
            DatasetType = 2;  %reset DatasetType to fit for the new subset
        end

    case 'Cal7'
        load('../dataset_Large-Scale/Caltech101-7.mat');
        X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;       
        k = numel(X0);
        
    case 'Cal20'
        load('../dataset_Large-Scale/Caltech101-20.mat');
        X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;       
        k = numel(X0);
        
    case 'HW'
        load('../dataset_Large-Scale/handwritten.mat');
        X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;       
        k = numel(X0);
        Y = Y + 1;
        
    case 'AWA4000'
        load('../Animals_with_Attributes/AWA4000/AWA4000.mat');
        k = 3; %choose first 3 features
        X0 = X(1:3); clear X; 
        
    case 'Yeast' % not a multi view dataset
        fid = fopen('../yeast/yeast.data','r');
        tmp = textscan(fid,'%s %f %f %f %f %f %f %f %f %s');
        fclose(fid);
        name = tmp{1};
        X = cell2mat(tmp(2:9))';
        class = tmp{10};
        [class_list, ~, Y] = unique(class);
        
    case 'USAA'
            file_path = '..\USAA.mat';
            load(file_path);
            
            Ytrn = train_video_label;
            Ytst = test_video_label;
        
        if strcmp(dataset_name_full, 'USAA')
            
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
            X0 = Data;
            k = numel(X0);
            clear Data;
            
        end 
        
    case 'AWA_MDR'
        filepath = '../computed_data/AWA_MDR.mat';
        load(filepath);
        X0 = Data; clear Data
        k = numel(X0);
        [~, ~, Y, ~] = ReadDataSetAWA;
        
    case 'USPS'
        filepathname = '../usps/usps.xlsx';
        tmp = xlsread(filepathname);
        Ytrn = tmp(:,1);
        Xtrn = tmp(:,3:2:end);
        
        filepathname = '../usps/usps_t.xlsx';
        tmp = xlsread(filepathname);
        Ytst = tmp(:,1);
        Xtst = tmp(:,3:2:end);
        X0 = [Xtrn',Xtst'];

end

%% ------------get label-----------
%k = numel(X0);
if DatasetType == 1
  label = [Ytrn; Ytst];
elseif DatasetType == 2
  label = Y;
end

if size(label,2) >1
  [label,~,~] = find(label'); %change label matrix into column
end

%% ------------normalization-----------
if iscell(X0)
    [~, nc] = cellfun(@size, X0);
    n = nc(1); clear nc;
    for i = 1:k
        X{i} = (X0{i}-repmat(min(X0{i},[],2),1,n))./...
            repmat(max(X0{i},[],2) - min(X0{i},[],2),1,n);
        tmp = X{i}(~any(isnan(X{i}),2),:); %% there are a lot nan here in such as 6943 ?? what happens here
        X{i} = tmp;
    end
else
    n = size(X0, 2);
    X{1} = (X0-repmat(min(X0,[],2),1,n))./...
        repmat(max(X0,[],2) - min(X0,[],2),1,n);
    tmp = X{1}(~any(isnan(X{1}),2),:); %% there are a lot nan here in such as 6943 ?? what happens here
    X{1} = tmp;
end
data = X;


%% old snippet for read classification data
% c = size(Ytrn,2);
% ntrn = size(Ytrn,1);
% ntst = size(Ytst,1);
% n = ntrn+ntst;
% 
% for i = 1:k
%     X{i} = (X0{i}-repmat(min(X0{i},[],2),1,n))./...
%         repmat(max(X0{i},[],2) - min(X0{i},[],2),1,n);
%     Xtrn{i} = X{i}(:,1:ntrn);
%     Xtst{i} = X{i}(:,ntrn+1:n);
% end
% 
% data = X;
% label = [Ytrn; Ytst];
