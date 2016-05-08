clear 
clc

addpath('..\semantic_metrics');
addpath('..\NCestimation_V2');
addpath('..\gadget');
addpath('..\Project_X\code')

dataset_name = 'MSRCV1'; %'AWA','MSRCV1'

%circlesdata; data = data'; %load test double moon data
%gaussiandata

[data, label] = readClusterDataset(dataset_name);

switch dataset_name
    case 'MSRCV1'
        [label_ind,~,~] = find(label'); %change label matrix into column
    case 'ApAy'
        label_ind = label;
    case 'AWA'
        label_ind = label;
end

nbclusters = 7;  %nbclusters = 2, 7
func = 'gaussdist';
algochoices = 'kmean';
k = 20;
sigma = 3000; 
epsilon = 1000;
gamma1 = 1;
gamma2 = 1;
eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];


%% kmeans
allData = cell2mat(data')';
[clusters, center] = kmeans(allData, nbclusters);

%evaluation
[~,RI,~,~] = valid_RandIndex(label_ind, clusters);
MIhat = MutualInfo(label_ind, clusters);
disp(RI);
disp(MIhat);

%% spectral clustering
[clusters, evalues, evectors] = spcl(data, nbclusters, sigma, 'sym', 'kmean', eigv);

%evaluation
[~,RI,~,~] = valid_RandIndex(label_ind, clusters);
MIhat = MutualInfo(label_ind, clusters);
disp(RI);
disp(MIhat);

%% multi-graph joint spectral clustering
%[C, obj_value, data_clustered] = cl_mg(data, nbclusters, {sigma, sigma, epsilon}, 'sym', 'kmean', [1 28]); %***
[C, Y, obj_value, data_clustered] = cl_mg_v2(data, nbclusters, {sigma, [k sigma], epsilon}, 'sym', algochoices, eigv); %***
[Y_ind,~,~] = find(Y');  %change label matrix into column

%evaluation
[~,RI,~,~] = valid_RandIndex(label_ind, Y_ind);
MIhat = MutualInfo(label_ind, Y_ind);
disp(RI);
disp(MIhat);

%% multi-view 
% [clusters, obj_value, F_record] = multi_view_fusion(data, nbclusters, gamma1, gamma2); % do pca on data first
% [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
% MIhat = MutualInfo(label_ind, clusters);
% disp(RI);
% disp(MIhat);





