clear 
clc

dataset_name = 'MSRCV1';
nbclusters = 7;
func = 'gaussdist';
algochoices = 'kmean';
sigma = 1000; 
epsilon = 3000;
eigv = [2 2]; %eigv = [1 28];

[data, label] = readClusterDataset(dataset_name);

%[C, obj_value, data_clustered] = cl_mg(data, nbclusters, {sigma, sigma, epsilon}, 'sym', 'kmean', [1 28]); %***
[C, obj_value, data_clustered] = cl_mg_v2(data, nbclusters, {sigma, sigma, epsilon}, 'sym', 'kmean', [1 28]); %***

[clusters, evalues, evectors] = spcl(data, nbclusters, {sigma, sigma, epsilon}, 'sym', 'kmean', [1 28]);


