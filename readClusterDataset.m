% clustering dataset reading
% dataset_name: NUS_lite, MSRCV1

function [data, label] = readClusterDataset(dataset_name, varargin)

  if ismember(dataset_name, {'NUS_lite', 'MSRCV1', 'AWA', 'ApAy'});
    DatasetType = 1;
  elseif ismember(dataset_name, {'Cal7', 'Cal20', 'HW' ,'NUSWIDEOBJ', 'AWA4000'});
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

    case 'AWA'
        [X_train, X_test, Ytrn, Ytst] = ReadDataSetAWA;
         k = numel(X_train);
         for i = 1:k
            X0{i} = [X_train{i},X_test{i}];
         end

    case 'ApAy'
        [~, ~, Ytrn, Ytst] = ReadDataSetApAy;
        comp_data_file = ['../computed_data/feat_reduce_',dataset_name,'.mat'];
        load(comp_data_file);
        X_train = {feat_trn_red'};
        X_test = {feat_tst_red'};
         k = numel(X_train);
         for i = 1:k
            X0{i} = [X_train{i};X_test{i}];
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
        load('V:\lance\Animals_with_Attributes\AWA4000\AWA4000.mat');
        X0 = X; clear X; 
        k = numel(X0);
        
    case 'Yeast' % not a multi view dataset
        fid = fopen('../yeast/yeast.data','r');
        tmp = textscan(fid,'%s %f %f %f %f %f %f %f %f %s');
        fclose(fid);
        name = tmp{1};
        X = cell2mat(tmp(2:9))';
        class = tmp{10};
        [class_list, ~, Y] = unique(class);
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
[~, nc] = cellfun(@size, X0); 
n = nc(1); clear nc;
for i = 1:k
  X{i} = (X0{i}-repmat(min(X0{i},[],2),1,n))./...
  repmat(max(X0{i},[],2) - min(X0{i},[],2),1,n);
  tmp = X{i}(~any(isnan(X{i}),2),:);
  X{i} = tmp;
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
