%% ===
% this code is a stand-alone clustering evalution code for the output
% clustering result labels.  
%%
clear
clc
% check the two varibles (dataset_name, ClusterResultsFile) 
% to make sure they are on the same dataset.
dataset_name = 'Cal20_cnn_MDR512 '; 
ClusterResultsFile = 'results\result_158_Cal20_cnn_MDR512_3_20_SPCL_ave_3.mat';

% dataset_name = 'Cal20 '; 
% ClusterResultsFile = 'results\clusterResults_Cal20.mat';

[~, Y] = readClusterDataset(dataset_name);
nbclusters = numel(unique(Y));  

load(ClusterResultsFile);
clusterMeasure = clusterResults;
t = fieldnames(clusterResults);
nmethod = numel(t);

for i = 1:nmethod
    predY = eval(['clusterResults.',t{i}]);
    result = ClusteringMeasure(Y, predY);
    clusterMeasure.(t{i}) = result;
end

[filepath,name,ext] = fileparts(ClusterResultsFile);
save(fullfile(filepath,['standAlone',name]), 'clusterMeasure');


