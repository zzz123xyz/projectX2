clear
clc

addpath('..\semantic_metrics');
addpath('..\NCestimation_V2');
addpath('..\gadget');
addpath('..\Project_X\code')
addpath(genpath('..\Project_CLR\CLR_code'))
addpath('..\project_MVSC')
addpath('..\project_MMSC')
addpath('..\project_MVCSS')

dataset_name = 'MSRCV1'; %'AWA','MSRCV1'
[data, label] = readClusterDataset(dataset_name);

switch dataset_name
case 'MSRCV1'
  [label_ind,~,~] = find(label'); %change label matrix into column
case 'ApAy'
  label_ind = label;
case 'AWA'
  label_ind = label;
end

methods = {'MVCSS'};
%methods = {'kmeans', 'SPCL'};
%methods = {'kmeans', 'SPCL', 'MVSC', 'MMSC', 'MVCSS', 'MVG', 'CLR', 'MVMG'};
nbclusters = 7;  %nbclusters = 2, 7

fnPart = '';
for i = 1:numel(methods)
  fnPart = [fnPart,[methods{i},'_']];
end

name = dir('results/result_*');
k = numel(name);
ii = k+1;
OutputFile = ['results/result_',num2str(ii,'%03i'),'_',num2str(nbclusters),'_',fnPart,'.txt'];
fid = fopen(OutputFile, 'wt');
fprintf(fid,['methods:',fnPart,'\n']);
fprintf(fid,['number of clusters:',num2str(nbclusters),'\n']);

nmethod = numel(methods);
for i=1:nmethod
  method = methods{i};
  switch method
  case 'kmeans'
    %% kmeans
    disp(method);
    fprintf(fid, [method,'\n']);
    allData = cell2mat(data')';
    [clusters, center] = kmeans(allData, nbclusters);

    %evaluation
    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
    MIhat = MutualInfo(label_ind, clusters);
    disp(RI);
    disp(MIhat);
    result = ClusteringMeasure(label_ind, clusters);
    disp(['ACC, MIhat, Purity:',num2str(result)]);
    fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);

  case  'SPCL'
    %% spectral clustering
    disp(method);
    fprintf(fid, [method,'\n']);

    algochoices = 'kmean';
    sigma = 3000;
    eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
    fprintf(fid, 'algochoices: %s \n', algochoices);
    fprintf(fid, 'sigma: %d \n', sigma);
    fprintf(fid, 'eigv: %s \n', num2str(eigv));

    [clusters, evalues, evectors] = spcl(data, nbclusters, sigma, 'sym', algochoices, eigv);

    %evaluation
    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
    MIhat = MutualInfo(label_ind, clusters);
    disp(RI);
    disp(MIhat);
    result = ClusteringMeasure(label_ind, clusters);
    disp(['ACC, MIhat, Purity:',num2str(result)]);
    fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);

  case 'MVSC'
    %% MVSC
    % Li, Y., Nie, F., Huang, H., & Huang, J. (2015, January).
    % Large-Scale Multi-View Spectral Clustering via Bipartite Graph.
    % In AAAI (pp. 2750-2756).
    disp(method);
    fprintf(fid, [method,'\n']);

    nbSltPnt = 40;
    sigma = 10;
    k = 8;
    func = 'gaussdist';
    fprintf(fid, 'nbSltPnt: %d \n', nbSltPnt);
    fprintf(fid, 'sigma: %d \n', sigma);
    fprintf(fid, 'k: %d \n', k);
    fprintf(fid, 'func: %s \n', func);

    for t = 0.1:0.2:2
      gamma = 10^t; % gamma = 10; may need to be changed
      fprintf(fid, 'gamma: %d \n', gamma);

      [clusters0, ~, obj_value, nbData] = MVSC(data, nbclusters, nbSltPnt, k, gamma, ...
      func, sigma);
      clusters = clusters0(1:nbData);
      [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
      MIhat = MutualInfo(label_ind, clusters);
      disp(RI);
      disp(MIhat);

      result = ClusteringMeasure(label_ind, clusters);
      disp(['ACC, MIhat, Purity:',num2str(result)]);
      fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n']);
    end
    fprintf(fid, '\n');

  case 'MMSC'
    %% MMSC
    %Cai, Xiao, et al. "Heterogeneous image feature integration via multi-modal
    %spectral clustering." Computer Vision and Pattern Recognition (CVPR),
    %2011 IEEE Conference on. IEEE, 2011.
    disp(method);
    fprintf(fid, [method,'\n']);

    func = 'SelfTune';
    param = 8;
    fprintf(fid, 'func: %s \n', func);
    fprintf(fid, 'param: %d \n', param);
    for t = -2:0.2:2
      a = 10^t;
      fprintf(fid, 'a: %d \n', a);

      [clusters, obj_value] = MMSC(data, nbclusters, a, func, param);
      [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
      MIhat = MutualInfo(label_ind, clusters);
      disp(RI);
      disp(MIhat);

      result = ClusteringMeasure(label_ind, clusters);
      disp(['ACC, MIhat, Purity:',num2str(result)]);
      fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n']);
    end
    fprintf(fid, '\n');

  case 'MVCSS'
    %% MVCSS
    % Multi-View Clustering and Feature Learning via Structured Sparsity
    % Wang, Hua, Feiping Nie, and Heng Huang. "Multi-view clustering and feature learning via structured sparsity."
    % Proceedings of the 30th International Conference on Machine Learning (ICML-13). 2013.
    disp(method);
    fprintf(fid, [method,'\n']);

    gamma1 = 1;
    gamma2 = 1;
    fprintf(fid, 'gamma1: %d \n', gamma1);
    fprintf(fid, 'gamma2: %d \n', gamma2);

    [clusters, obj_value, F_record] = multi_view_fusion(data, nbclusters, gamma1, gamma2); % do pca on data first
    %evaluation
    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
    MIhat = MutualInfo(label_ind, clusters);
    disp(RI);
    disp(MIhat);

    result = ClusteringMeasure(label_ind, clusters);
    disp(['ACC, MIhat, Purity:',num2str(result)]);
    fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);

  case 'MVG'
    %% MVG
    % multi-view single graph joint clustering
    disp(method);
    fprintf(fid, [method,'\n']);

    m = 7;
    eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
    fprintf(fid, 'm: %d \n', m);
    fprintf(fid, 'eigv: %d \n', num2str(eigv));

    [C, Y, obj_value, data_clustered] = MVG(data, nbclusters, eigv, 'CLR', m); %***
    [clusters,~,~] = find(Y');  %change label matrix into column

    %evaluation
    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
    MIhat = MutualInfo(label_ind, clusters);
    disp(RI);
    disp(MIhat);

    result = ClusteringMeasure(label_ind, clusters);
    disp(['ACC, MIhat, Purity:',num2str(result)]);
    fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);

  case 'CLR'
    %% CLR
    % Nie, Feiping, et al. "The Constrained Laplacian Rank Algorithm for Graph-Based Clustering." (2016).
    disp(method);
    fprintf(fid, [method,'\n']);

    m = 7; % a para to tune m <10 in paper
    fprintf(fid, 'm: %d \n', m);

    AllDataMatrix = DataConcatenate(data);
    [clusters, S, evectors, cs] = CLR_main(AllDataMatrix, nbclusters, m);

    %evaluation
    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
    MIhat = MutualInfo(label_ind, clusters);
    disp(RI);
    disp(MIhat);

    result = ClusteringMeasure(label_ind, clusters);
    disp(['ACC, MIhat, Purity:',num2str(result)]);
    fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);

  case 'MVMG'
    %% MVMG
    % multi-graph joint spectral clustering
    %[C, obj_value, data_clustered] = cl_mg(data, nbclusters, {sigma, sigma, epsilon}, 'sym', 'kmean', [1 28]); %***
    disp(method);
    fprintf(fid, [method,'\n']);

    algochoices = 'kmean';
    sigma = 3000;
    epsilon = 1000;
    k = 20;
    fprintf(fid, 'algochoices: %d \n', algochoices);
    fprintf(fid, 'sigma: %d \n', sigma);
    fprintf(fid, 'epsilon: %d \n', epsilon);
    fprintf(fid, 'k: %d \n', k);

    [C, Y, obj_value, data_clustered] = cl_mg_v2(data, nbclusters, {sigma, [k sigma], epsilon, m}, 'sym', algochoices, eigv); %***
    [clusters,~,~] = find(Y');  %change label matrix into column

    %evaluation
    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
    MIhat = MutualInfo(label_ind, clusters);
    disp(RI);
    disp(MIhat);

    result = ClusteringMeasure(label_ind, clusters);
    disp(['ACC, MIhat, Purity:',num2str(result)]);
    fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);
  end

end

fclose(fid);
