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

% if nargin>1 %****** 
%    OOS_flag = varargin{1};
%    OOS_ratio = varargin{2};
% end

tmp = strfind(dataset_name_full,'_');
if isempty(tmp)
    dataset_name = dataset_name_full;
else
    dataset_name = dataset_name_full(1:tmp(1)-1);
end

if ismember(dataset_name, {'NUSlite', 'MSRCV1', 'AWA', 'ApAy', 'USAA', 'USPS'})
    DatasetType = 1;
elseif ismember(dataset_name, {'Cal7', 'Cal20', 'HW' ,'NUSWIDEOBJ', 'AWA4000'...
        ,'ApAy_MDR','AWA_MDR', 'Coil20', 'recommendationM', 'recommendationO'})
    DatasetType = 2;
end

switch dataset_name
    case 'NUSlite'
        addpath(genpath('../Project_X/code'))
        label_path = '../Project_X/NUS_WIDE_lite/NUS-WIDE-SCENE/ground truth/';
        train_path = {'../Project_X/NUS_WIDE_lite/NUS-WIDE-SCENE/low level features/Train_CH.txt',...
                    '../Project_X/NUS_WIDE_lite/NUS-WIDE-SCENE/low level features/Train_EDH.txt',...
                    '../Project_X/NUS_WIDE_lite/NUS-WIDE-SCENE/low level features/Train_WT.txt'};
        test_path = {'../Project_X/NUS_WIDE_lite/NUS-WIDE-SCENE/low level features/Test_CH.txt',...
                     '../Project_X/NUS_WIDE_lite/NUS-WIDE-SCENE/low level features/TEST_EDH.txt',...
                     '../Project_X/NUS_WIDE_lite/NUS-WIDE-SCENE/low level features/Test_WT.txt'};

        data_path = {train_path,test_path};
        k = numel(train_path);
 
        Ytrn = double(ReadLabel(label_path, 'Train'));
        Ytst = double(ReadLabel(label_path, 'Test'));
        label = [Ytrn; Ytst];

        nconcept = size(Ytrn,2);

        for i = 1:k
            X0{i} = [ReadData(train_path{i}),ReadData(test_path{i})];
            kf(i) = size(X0{i},1);
        end
    case 'NUSWIDEOBJ'    
        load('../dataset_Large-Scale/NUSWIDEOBJ.mat');
        X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;       
        k = numel(X0);
        label = Y;
        
    case 'MSRCV1'
        load('../Project_X/MSRCV1.mat');

        Ytrn = Y_train;
        Ytst = Y_test;
        label = [Ytrn; Ytst];

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

    case 'ASUNfull'
        %=================Get classes vector for samples=====================
        images_file = '../SUN/SUNAttributeDB/images.mat'; 
        load(images_file); %load images(image names)

        nsamples = numel(images);
        label = zeros(nsamples,1);
        expression = '/';

        for i = 1:nsamples
            image_name = images{i};
            startIndex = regexp(image_name,expression);
            image_class{i} = image_name(1:startIndex(end));
        end

        class_name = unique(image_class);
        nclass = numel(class_name);

        for i = 1:nsamples
            for j = 1:nclass
                if strcmp(image_class(i),class_name(j));
                    label(i) = j;
                end
            end
        end

        load('../SUN/data_v2/geo_color_image_features.mat'); % geo_color feat
        X0{1} = feature_vector';
        load('../SUN/data_v2/ssim_image_features.mat'); % ssim feat
        X0{2} = feature_vector';
        load('../SUN/data_v2/hog2x2_image_features.mat'); % hog2x2 feat
        X0{3} = feature_vector';
        load('../SUN/data_v2/gist_image_features.mat'); % gist feat
        X0{4} = feature_vector'; % load all feature_vector
        k = numel(X0);

    case 'AWA'
        [X_train, X_test, Ytrn, Ytst] = ReadDataSetAWA;
         k = numel(X_train);
         for i = 1:k
            X0{i} = [X_train{i},X_test{i}];
         end

         label = [Ytrn; Ytst];

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
            label = [Ytrn; Ytst];
            
        elseif strcmp(dataset_name_full, 'ApAy_4_trn')    
            [~, ~, Y, ~] = ReadDataSetApAy;
            filepath = ['../computed_data/',dataset_name_full];
            load(filepath);  %load Data
            X0 = Data; clear Data
            k = numel(X0);
            DatasetType = 2;
            label = Y;

        else
            [~, ~, Y, ~] = ReadDataSetApAy; % only consider training samples
            filepath = ['../computed_data/',dataset_name_full];
            load(filepath); %load Data, testIndicator, trainIndicator
            X0 = Data; clear Data
            k = numel(X0);
            DatasetType = 2;  %reset DatasetType to fit for the new subset
            label = Y;
        end

    case 'Cal7'
        load('../dataset_Large-Scale/Caltech101-7.mat');
        X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;       
        k = numel(X0);
        label = Y;
        
    case 'Cal20'
        load('../dataset_Large-Scale/Caltech101-20.mat');
        X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;       
        k = numel(X0);
        label = Y;
        
    case 'HW'
        load('../dataset_Large-Scale/handwritten.mat');
        X0 = cellfun(@transpose, X, 'UniformOutput', false); clear X;       
        k = numel(X0);
        Y = Y + 1;
        label = Y;
        
    case 'AWA4000'
        load('../Animals_with_Attributes/AWA4000/AWA4000.mat');
        k = 3; %choose first 3 features
        X0 = X(1:3); clear X; 
        label = Y;
        
    case 'Yeast' % not a multi view dataset
        fid = fopen('../yeast/yeast.data','r');
        tmp = textscan(fid,'%s %f %f %f %f %f %f %f %f %s');
        fclose(fid);
        name = tmp{1};
        X = cell2mat(tmp(2:9))';
        class = tmp{10};
        [class_list, ~, Y] = unique(class);
        label = Y;

    case 'USAA'
        file_path = '../USAA.mat';
        load(file_path);

        Ytrn = train_video_label;
        Ytst = test_video_label;
        label = [Ytrn; Ytst];
        
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
            file_path = ['../computed_data/',dataset_name_full];
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
        label = Y;
        
    case 'USPS'
        filepathname = '../usps/usps.xlsx';
        tmp = xlsread(filepathname);
        Ytrn = tmp(:,1);
        Xtrn = tmp(:,3:2:end);
        label = [Ytrn; Ytst];
        
        filepathname = '../usps/usps_t.xlsx';
        tmp = xlsread(filepathname);
        Ytst = tmp(:,1);
        Xtst = tmp(:,3:2:end);
        X0 = [Xtrn',Xtst'];
        
    case 'Coil20'
        filepathname = dir('../dataset_Large-Scale/new_dataset/coil-20-proc/*.png');
        % get the path to the coil20 dataset
%         filepathname = extractfield(filepathname,'name'); % can use this
%        when there is mapping tool
        filepathname = {filepathname.name};
        Y = cellfun(@(x) x(strfind(x,'j')+1:strfind(x,'_')-1), filepathname, 'UniformOutput', false);
        % use celfun to find the number between j and _ as the label 
        % note the strfind(x,'_') only return the first index of '_'
        Y = cellfun(@str2num, Y)';
        label = Y;
        
    case 'recommendationM'
        load('../dataset_Large-Scale/new_dataset/recommendation/musical_instrument/user.mat') %load user data
        X0{1}=data;
        load('../dataset_Large-Scale/new_dataset/recommendation/musical_instrument/item.mat') %load user data
        X0{2}=data;
        k = numel(X0);
        
        % synthesize a random integer label vector cuz there are no groudtruth label in this
        % situation
        nsample = size(data, 2);
        Y = randi(10, 1, nsample)';
        label = Y;
        
    case 'recommendationO'
        load('../dataset_Large-Scale/new_dataset/recommendation/Office_Products/user.mat') %load user data
        X0{1}=data;
        load('../dataset_Large-Scale/new_dataset/recommendation/Office_Products/item.mat') %load user data
        X0{2}=data;
        k = numel(X0);
        
        % synthesize a random integer label vector cuz there are no groudtruth label in this
        % situation
        nsample = size(data, 2);
        Y = randi(10, 1, nsample)';
        label = Y;

    case 'youtubeface'
        
end

%% ------------change label matrix into column-----------
if size(label,2) >1
  [label,~,~] = find(label'); 
end

%% ------------normalization-----------
if exist('X0', 'var')
    if iscell(X0)
        [~, nc] = cellfun(@size, X0);
        n = nc(1); clear nc;
        for i = 1:k  %k is the number of views
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
else 
    data = [];
end

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
