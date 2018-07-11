%% ===
% this code is a stand-alone clustering evalution code for the output
% clustering result labels.
%%
clear
clc

addpath('..\NCestimation_V2');
% check the two varibles (dataset_name, ClusterResultsFile)
% to make sure they are on the same dataset.
% dataset_name = 'Cal20_cnn_MDR512 ';
% ClusterResultsFile = 'results\result_158_Cal20_cnn_MDR512_3_20_SPCL_ave_3.mat';
% default

dataset_name = 'HW';
ClusterResultsFile = ['results\','result_211_HW_all_10_kmeans_SPCL_SPCLNaive_MVSC_MMSC_MVCSS_MVG_CLR_MVMG_ave_3'];
nreps = str2double(ClusterResultsFile(end));  %nreps = str2double(ClusterResultsFile(end));default
nmeasure = 6;
flag = 0; %if process the new type of results flag = 1 uncompleted;

% dataset_name = 'Cal20 ';
% ClusterResultsFile = 'results\clusterResults_Cal20.mat';

[~, Y] = readClusterDataset(dataset_name);
nbclusters = numel(unique(Y));

load(ClusterResultsFile); %load clusterResults
%clusterMeasure = clusterResults;
tmp1 = fieldnames(clusterResults);

%% determine if the field in clusterResults a raw data recoard or measure result
t = [];
for i = 1:numel(tmp1)
    if ~iscell(clusterResults.(tmp1{i}))
       t =[t; tmp1(i)];
    end
end
nmethod = numel(t);

%% uncompleted to process new type of results
% if sum(contains(tmp1,'measure'))>=1
%    flag = 1;  
% end
% if flag == 0
    
for i = 1:nmethod
    predY = eval(['clusterResults.',t{i}]);
    
    if size(predY,2) == 1
        result = ClusteringMeasure(Y, predY);
        clusterMeasure.(t{i}) = result;
    else
        tmp = 0:nreps:size(predY,2);
        edpoint = tmp(2:end);
        stpoint = tmp(1:end-1)+1;
        
        result = [];
        resultAVE = zeros(numel(stpoint),nmeasure);
        resultSTD = zeros(numel(stpoint),nmeasure);
        for j = 1:numel(stpoint)
            resultSinglePara = [];
            for k = stpoint(j):edpoint(j)
                resultSinglePara = [resultSinglePara; ClusteringMeasure(Y, predY(:,k))];
            end
            result{j} = resultSinglePara;
            resultAVE(j,:) = mean(resultSinglePara);
            resultSTD(j,:) = std(resultSinglePara, 0, 1);
        end
        result = result';
        clusterMeasure.(t{i}) = result;
        clusterMeasure.([t{i},'_AVE']) = resultAVE;
        clusterMeasure.([t{i},'_STD']) = resultSTD;
    end
end

[filepath,name,ext] = fileparts(ClusterResultsFile);
save(fullfile(filepath,['standAlone',name]), 'clusterMeasure');


