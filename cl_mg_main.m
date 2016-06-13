clear 
clc

addpath('..\semantic_metrics');
addpath('..\NCestimation_V2');
addpath('..\gadget');
addpath('..\Project_X\code')
addpath(genpath('..\Project_CLR\CLR_code'))
addpath('..\project_MVSC')
addpath('..\project_MMSC')

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

%% MVSC
for t = 0.1:0.2:2
    nbSltPnt = 40;
    gamma_mvsc = 10^t; % gamma_mvsc = 10; may need to be changed
    sigma_mvsc = 10;
    k_mvsc = 8;
    [clusters0, ~, obj_value, nbData] = MVSC(data, nbclusters, nbSltPnt, k_mvsc, gamma_mvsc, ...
        func, sigma_mvsc);
    clusters = clusters0(1:nbData);
    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
    MIhat = MutualInfo(label_ind, clusters);
    disp(RI);
    disp(MIhat);
end

%% MMSC
for t = -2:0.2:2
    a_MMSC = 10^t;
    fun_MMSC = 'SelfTune';
    param_MMSC = 8;
    [clusters, obj_value] = MMSC(data, nbclusters, a_MMSC, fun_MMSC, param_MMSC);
    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
    MIhat = MutualInfo(label_ind, clusters);
    disp(RI);
    disp(MIhat);
end

%% multi-view 
[clusters, obj_value, F_record] = multi_view_fusion(data, nbclusters, gamma1, gamma2); % do pca on data first
[~,RI,~,~] = valid_RandIndex(label_ind, clusters);
MIhat = MutualInfo(label_ind, clusters);
disp(RI);
disp(MIhat);

%% multi-view single graph joint clustering
m = 7;
[C, Y, obj_value, data_clustered] = MVG(data, nbclusters, eigv, 'CLR', m); %***
[Y_ind,~,~] = find(Y');  %change label matrix into column

%evaluation
[~,RI,~,~] = valid_RandIndex(label_ind, Y_ind);
MIhat = MutualInfo(label_ind, Y_ind);
disp(RI);
disp(MIhat);


%% CLR
m = 7; % a para to tune m <10 in paper
AllDataMatrix = DataConcatenate(data);
[clusters, S, evectors, cs] = CLR_main(AllDataMatrix, nbclusters, m);

%evaluation
[~,RI,~,~] = valid_RandIndex(label_ind, clusters);
MIhat = MutualInfo(label_ind, clusters);
disp(RI);
disp(MIhat);

%% multi-graph joint spectral clustering
%[C, obj_value, data_clustered] = cl_mg(data, nbclusters, {sigma, sigma, epsilon}, 'sym', 'kmean', [1 28]); %***
[C, Y, obj_value, data_clustered] = cl_mg_v2(data, nbclusters, {sigma, [k sigma], epsilon, m}, 'sym', algochoices, eigv); %***
[Y_ind,~,~] = find(Y');  %change label matrix into column

%evaluation
[~,RI,~,~] = valid_RandIndex(label_ind, Y_ind);
MIhat = MutualInfo(label_ind, Y_ind);
disp(RI);
disp(MIhat);









