clear
clc

addpath('..\semantic_metrics');
addpath('..\NCestimation_V2');
addpath('..\gadget');
%addpath('..\Project_X\code')
addpath(genpath('..\Project_CLR\CLR_code'))
addpath('..\project_MVSC')
addpath('..\project_MMSC')
addpath('..\project_MVCSS')
addpath('..\clustering_eval_kun')
addpath('..\ApAy_dataset')
addpath('..\Animals_with_Attributes')
addpath('ONGC')

%% ==== dataset and global para ==== %!!!!
%dataset_name = 'MSRCV1'; %'AWA','MSRCV1','NUSWIDEOBJ','Cal7','Cal20',
%'HW',AWA4000,'ApAy','AWA_MDR','ApAy_MDR'(rename from
%ApAy_MDR1,ApAy_MDR2... ),ApAy_MDR_R01R01R005, A
%'USAA','USAA_MDR_R005', Cal20_cnn, Cal20_cnn_MDR512, ApAy_cnn_MDR512;
% ApAy_trn_cnn_MDR512
dataset_name = 'ApAy_trn_cnn_MDR512'; 
featType = 'all'; % default : all
nreps = 5; % parameter  default : 1
clusterResults = struct;
%bestClusterPara = getBestPara(dataset_name); % using best parameters if
%there any !!!
%saveClusterResultsFile = ['results\clusterResults_',dataset_name,'.mat'];

%% ==== selecting methods ==== %!!!!
%methods = {'kmeans','SPCL','SPCLNaive','MVSC','MMSC','MVCSS','MVG','CLR','MVMG'}; % default
%methods = {'kmeans'};
%methods = {'SPCL'}; % for single view
%methods = {'SPCLNaive'};
%methods = {'MVSC'};
%methods = {'MMSC'};
%methods = {'MVCSS'};
%methods = {'MVG'};
%methods = {'CLR'};
%methods = {'MVMG'};
%methods = {'SPCL', 'MVSC', 'MVG', 'CLR', 'MVMG'}; %chosen
%methods = {'kmeans', 'SPCL', 'MVSC', 'MMSC', 'MVCSS', 'MVG', 'CLR', 'MVMG'};
methods = {'ONGC'};

%% ==== read dataset ====
[~, label_ind] = readClusterDataset(dataset_name);
data = readDeepFeat(dataset_name, featType);

nbclusters = numel(unique(label_ind));  %nbclusters =  7, 
nmethod = numel(methods);

%% ==== name the save files ====
fnPart = '';
for i = 1:numel(methods)
  fnPart = [fnPart,[methods{i},'_']];
end

name = dir('results/result_*');
k = numel(name);
ii = k+1;
OutputFile = ['results/result_',num2str(ii,'%03i'),'_',dataset_name,'_',featType,'_',num2str(nbclusters),'_',fnPart,'ave_',num2str(nreps),'.txt'];
saveClusterResultsFile = ['results/result_',num2str(ii,'%03i'),'_',dataset_name,'_',featType,'_',num2str(nbclusters),'_',fnPart,'ave_',num2str(nreps),'.mat'];

try 
fid = fopen(OutputFile, 'wt');
fprintf(fid,['filename:',OutputFile,'\n']);
fprintf(fid,['dataset:',dataset_name,'\n']);
fprintf(fid,['number of clusters:',num2str(nbclusters),'\n']);
fprintf(fid,['number of repeats:',num2str(nreps),'\n']);
fprintf(fid,['methods:',fnPart,'\n\n']);

%% ==== loop start ====
for i=1:nmethod
    method = methods{i};
    switch method
        case 'kmeans'
            %% kmeans
            clusterResults.kmeans = []; %initialize clusterResults.kmeans
            disp(method);
            fprintf(fid, [method,'\n']);
            if iscell(data)
                allData = cell2mat(data')';
            else
                allData = data;
            end
             
            allResults = zeros(nreps,4);
            
            for v = 1:nreps
                [clusters, center] = kmeans(allData, nbclusters);
                clusterResults.kmeans = [clusterResults.kmeans, clusters];
                %evaluation
                [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                MIhat = MutualInfo(label_ind, clusters);
                disp(RI);
                disp(MIhat);
                singleResult = ClusteringMeasure(label_ind, clusters);
                allResults(v,:) = singleResult;

            end
            result = mean(allResults,1); % result is average result;
            
            disp(['ACC, MIhat, Purity, F1score: ',num2str(result)]);
            fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
            
        case  'SPCL'
           %% spectral clustering
            clusterResults.SPCL = []; %initialize clusterResults.SPCL
            disp(method);
            fprintf(fid, [method,'\n']);
            
            algochoices = 'kmeans';
            fprintf(fid, 'algochoices: %s \n', algochoices);
            
            eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
            fprintf(fid, 'eigv: %s \n\n', num2str(eigv));
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.SPCL.sigma ; %the predefine sigma parameter 2.7 MSCRv1;
                %the predefine sigma parameter 4.32 Cal20;
            end
            
            for percent = 0.05: 0.05: 0.5 % the following if block can not used for this line, consider to change ***
                %percent = 0.1: 0.05: 0.5 % default !!!
                
                sigma = determineSigma(data, 1, percent);
                
                % if there is a predefine parameter
                if exist('pdpara','var')&&percent==0.05
                    sigma = pdpara;
                elseif  exist('pdpara','var')&&percent>0.05
                    clear pdpara;
                    break;
                end
                
                fprintf(fid, 'sigma: %f \n', sigma);
                
                allResults = zeros(nreps,4);
                for v = 1:nreps
                    [clusters, evalues, evectors] = spcl...
                        (data, nbclusters, sigma, 'sym', algochoices, eigv);
                    clusterResults.SPCL = [clusterResults.SPCL, clusters];
                    %evaluation
                    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                    MIhat = MutualInfo(label_ind, clusters);
                    disp(RI);
                    disp(MIhat);
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                    
                end
                result = mean(allResults,1); % result is average result;
                
                disp(['ACC, MIhat, Purity, F1score: ',num2str(result)]);
                fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
            end
            
        case 'SPCLNaive'
           %% spectral clustering Naive mode (apply the spectral clustering
            % algorithm on the combined Laplacian matrix which is
            % the summation of five Laplacian matrix corresponding
            % to each single modal)
            
            clusterResults.SPCLNaive = []; %initialize clusterResults.SPCLNaive
            disp(method);
            fprintf(fid, [method,'\n']);
            
            func = 'gaussdist';
            fprintf(fid, 'fun: %s \n', func);
            
            algochoices = 'kmeans';
            fprintf(fid, 'algochoices: %s \n', algochoices);
            
            eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
            fprintf(fid, 'eigv: %s \n', num2str(eigv));
            %sigma = 3000;

            %                 sigma = determineSigma(data, 1, percent);
            %                 fprintf(fid, 'sigma: %f \n', sigma);
            allResults = zeros(nreps,4);
            for v = 1:nreps
                [clusters, evalues, evectors] = spclNaive...
                    (data, nbclusters, func, 'sym', algochoices, eigv);
                clusterResults.SPCLNaive = [clusterResults.SPCLNaive, clusters];
                %evaluation
                [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                MIhat = MutualInfo(label_ind, clusters);
                disp(RI);
                disp(MIhat);
                singleResult = ClusteringMeasure(label_ind, clusters);
                allResults(v,:) = singleResult;
                
            end
            result = mean(allResults,1); % result is average result;
                          
            disp(['ACC, MIhat, Purity, F1score: ',num2str(result)]);
            fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
            
        case 'MVSC'
            %% MVSC
            % Li, Y., Nie, F., Huang, H., & Huang, J. (2015, January).
            % Large-Scale Multi-View Spectral Clustering via Bipartite Graph.
            % In AAAI (pp. 2750-2756).
            
            clusterResults.MVSC = []; %initialize clusterResults.MVSC
            disp(method);
            fprintf(fid, [method,'\n']);
            
            if exist('bestClusterPara','var')
                nbSltPnt = bestClusterPara.MVSC.nbSltPnt; %nbSltPnt = 40 for MSCRV1 %400  others
            else
                nbSltPnt = 400;
            end
            
            percent = 0.35;
            sigma = determineSigma(data, 1, percent); % assume the value is as the same as in SPCL
            k = 8;
            func = 'gaussdist';
            fprintf(fid, 'nbSltPnt: %d \n', nbSltPnt);
            fprintf(fid, 'sigma: %f \n', sigma);
            fprintf(fid, 'k: %d \n', k);
            fprintf(fid, 'func: %s \n\n', func);
            param_list = 0.1:0.2:2;
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.MVSC.gamma;
            end
            
            for j = 1:numel(param_list)
                
                t = param_list(j);
                gamma = 10^t; % gamma = 10; may need to be changed
                
                % if there is a predefine parameter
                if exist('pdpara','var')&&j==1
                    gamma = pdpara;
                elseif  exist('pdpara','var')&&j>1
                    clear pdpara;
                    break;
                end
                
                fprintf(fid, 'gamma: %f \n', gamma);
                
                allResults = zeros(nreps,4);
                for v = 1:nreps
                    
                    [clusters0, ~, obj_value, nbData] = MVSC...
                        (data, nbclusters, nbSltPnt, k, gamma, func, sigma);
                    clusters = clusters0(1:nbData);
                    clusterResults.MVSC = [clusterResults.MVSC, clusters];
                    
                    %evaluation
                    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                    MIhat = MutualInfo(label_ind, clusters);
                    disp(RI);
                    disp(MIhat);
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                end
                result = mean(allResults,1); % result is average result;
                
                disp(['ACC, MIhat, Purity, F1score: ',num2str(result)]);
                fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n']);
            end
            fprintf(fid, '\n');
            
        case 'MMSC'
            %% MMSC
            %Cai, Xiao, et al. "Heterogeneous image feature integration via multi-modal
            %spectral clustering." Computer Vision and Pattern Recognition (CVPR),
            %2011 IEEE Conference on. IEEE, 2011.
            
            clusterResults.MMSC = []; %initialize clusterResults.MMSC
            disp(method);
            fprintf(fid, [method,'\n']);
            
            func = 'SelfTune';
            %func = 'gaussdist';
            param = 10;  %in paper it's k=9 in P5, the first col is datapoints themselves
            %discrete_model = 'rotation';
            discrete_model = 'rotation';
            
            fprintf(fid, 'func: %s \n', func);
            fprintf(fid, 'param: %d \n', param);
            fprintf(fid, 'discrete_model: %s \n\n', discrete_model);
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.MMSC.a;
            end
            
            for t = -2:0.2:2
                a = 10^t;
                
                % if there is a predefine parameter
                if exist('pdpara','var')&&t==-2
                    a = pdpara;
                elseif  exist('pdpara','var')&&t>-2
                    clear pdpara;
                    break;
                end
                
                fprintf(fid, 'a: %f \n', a);
                
                allResults = zeros(nreps,4);
                for v = 1:nreps
                    
                    %[clusters, obj_value] = MMSC(data, nbclusters, a, func, param);
                    Y = MMSC_main(data, nbclusters, a, func, param, discrete_model);
                    
                    if strcmp(discrete_model,'nmf')
                        clusters = kmeans(Y, nbclusters);
                    else
                        [clusters,~,~] = find(Y');  %change label matrix into column
                    end
                    
                    clusterResults.MMSC = [clusterResults.MMSC, clusters];
                    
                    %evaluation
                    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                    MIhat = MutualInfo(label_ind, clusters);
                    disp(RI);
                    disp(MIhat);
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                end
                result = mean(allResults,1); % result is average result;
                
                disp(['ACC, MIhat, Purity, F1score: ',num2str(result)]);
                fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n']);
            end
            fprintf(fid, '\n');
            
        case 'MVCSS'
            %% MVCSS
            % Multi-View Clustering and Feature Learning via Structured Sparsity
            % Wang, Hua, Feiping Nie, and Heng Huang. "Multi-view clustering and feature learning via structured sparsity."
            % Proceedings of the 30th International Conference on Machine Learning (ICML-13). 2013.
            
            clusterResults.MVCSS = []; %initialize clusterResults.MVCSS
            disp(method);
            fprintf(fid, [method,'\n\n']);
            
            % exponent = [-5 : 5];
            exponent = 1;
            
            if exist('bestClusterPara','var')
                pdpara1 = bestClusterPara.MVCSS.gamma1;
                pdpara2 = bestClusterPara.MVCSS.gamma2;
            end
            
            for i = 1: numel(exponent)
                gamma1 = 10^exponent(i);
                
                 % if there is a predefine parameter
                if exist('pdpara1','var')&&i==1
                    gamma1 = pdpara1;
                elseif  exist('pdpara1','var')&&i>1
                    clear pdpara1;
                    break;
                end
                
                for j =  1: numel(exponent)
                    gamma2 = 10^exponent(j);
                    
                    % if there is a predefine parameter
                    if exist('pdpara2','var')&&j==1
                        gamma2 = pdpara2;
                    elseif  exist('pdpara2','var')&&j>1
                        clear pdpara2;
                        break;
                    end
                    
                    fprintf(fid, 'gamma1: %d \n', gamma1);
                    fprintf(fid, 'gamma2: %d \n', gamma2);
                    
                    allResults = zeros(nreps,4);
                    for v = 1:nreps
                        [clusters, obj_value, F_record] = multi_view_fusion...
                            (data, nbclusters, gamma1, gamma2); % do pca on data first, No you can use pinv, right?
                        clusterResults.MVCSS = [clusterResults.MVCSS, clusters];
                        
                        %evaluation
                        [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                        MIhat = MutualInfo(label_ind, clusters);
                        disp(RI);
                        disp(MIhat);
                        singleResult = ClusteringMeasure(label_ind, clusters);
                        allResults(v,:) = singleResult;
                    end
                    result = mean(allResults,1); % result is average result;
                    
                    disp(['ACC, MIhat, Purity, F1score: ',num2str(result)]);
                    fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                end
            end
            
            
        case 'MVG'
            %% MVG
            % multi-view single graph joint clustering
            
            clusterResults.MVG = []; %initialize clusterResults.MVG
            disp(method);
            fprintf(fid, [method,'\n']);
            
            m = 9;
            if strncmpi(dataset_name,'ApAy_MDR',8)
                m = [9, 9, 30];
            elseif  strncmpi(dataset_name,'USAA',4)
                m = 8;
            end
            
            eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
            fprintf(fid, 'm: %d \n', m);
            fprintf(fid, 'eigv: %s \n\n', num2str(eigv));
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.MVG.eta;
            end
            
            maxResult = -inf; %predefine the varible to save the highest performance
            for t = -2:0.2:2
                eta = 10^t;
                
                % if there is a predefine parameter
                if exist('pdpara','var')&&t==-2
                    eta = pdpara;
                elseif  exist('pdpara','var')&&t>-2
                    clear pdpara;
                    break;
                end
                
                fprintf(fid, 'eta: %f \n', eta);
                
                allResults = zeros(nreps,4);
                for v = 1:nreps
                    [C, Y, obj_value, data_clustered] = MVG(data, nbclusters, eta, eigv, 'CLR', m); %***
                    [clusters,~,~] = find(Y');  %change label matrix into column
                    clusterResults.MVG = [clusterResults.MVG, clusters];
                    
                    %evaluation
                    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                    MIhat = MutualInfo(label_ind, clusters);
                    disp(RI);
                    disp(MIhat);
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                    %obtain the clusters results from highest performance
                    if mean(singleResult) > maxResult
                        maxResult = mean(singleResult);
                        clusterBestResults.MVG.result = clusters;
                        clusterBestResults.MVG.para.eta = eta;
                    end
                end
                result = mean(allResults,1); % result is average result;
                
                disp(['ACC, MIhat, Purity, F1score: ',num2str(result)]);
                fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
            end
            
            fprintf(fid, '\n');
            
        case 'CLR'
            %% CLR
            % Nie, Feiping, et al. "The Constrained Laplacian Rank Algorithm for Graph-Based Clustering." (2016).
            
            clusterResults.CLR = []; %initialize clusterResults.CLR
            disp(method);
            fprintf(fid, [method,'\n\n']);
            
            %m = 7; % a para to tune m <10 in paper
            AllDataMatrix = DataConcatenate(data);
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.CLR.m;
            end
            
            for m = 2:10 % a para to tune m <10 in paper, right when perform on cal20
            %for m = 4:7 % a para to tune m <10 in paper, right when perform on cal20
            
                % if there is a predefine parameter
                if exist('pdpara','var')&&m==2
                    m = pdpara;
                elseif  exist('pdpara','var')&&m>2
                    clear pdpara;
                    break;
                end
                
                allResults = zeros(nreps,4);
                
                flag = 0;
                
                while flag == 0
                    try
                        for v = 1:nreps
                            [clusters, S, evectors, cs] = CLR_main(AllDataMatrix, nbclusters, m);
                            clusterResults.CLR = [clusterResults.CLR, clusters];
                            
                            %evaluation
                            [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                            MIhat = MutualInfo(label_ind, clusters);
                            disp(RI);
                            disp(MIhat);
                            singleResult = ClusteringMeasure(label_ind, clusters);
                            allResults(v,:) = singleResult;
                            
                            flag = 1;
                        end
                    catch
                        warning('Problem: set new m value because nbclusters is less than number of connected components');
                        m = m + 1;
                    end
                    
                end
                
                result = mean(allResults,1); % result is average result;
                
                fprintf(fid, 'm: %d \n', m);
                disp(['ACC, MIhat, Purity, F1score: ',num2str(result)]);
                fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
            end
            fprintf(fid, '\n');
            
        case 'MVMG'
            %% MVMG
            % multi-graph joint spectral clustering
            %[C, obj_value, data_clustered] = cl_mg(data, nbclusters, {sigma, sigma, epsilon}, 'sym', 'kmeans', [1 28]); %***
            
            clusterResults.MVMG = []; %initialize clusterResults.MVMG
            disp(method);
            fprintf(fid, [method,'\n']);
            
            algochoices = 'kmeans';
            %sigma = 3000;
            m = 9; % the best setting from experiment in CLR
            sigma = determineSigma(data, 1, 0.15);
            epsilon = sigma;
            k = 20;
            eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
            
            fprintf(fid, 'algochoices: %s \n', algochoices);
            fprintf(fid, 'sigma: %f \n', sigma);
            fprintf(fid, 'epsilon: %f \n', epsilon);
            fprintf(fid, 'k: %d \n', k);
            fprintf(fid, 'eigv: %s \n\n', num2str(eigv));
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.MVMG.eta;
            end
            
            maxResult = -inf; %predefine the varible to save the highest performance
            for t = -2:0.2:2
                eta = 10^t;
                
                % if there is a predefine parameter
                if exist('pdpara','var')&&t==-2
                    eta = pdpara;
                elseif  exist('pdpara','var')&&t>-2
                    clear pdpara;
                    break;
                end
                
                fprintf(fid, 'eta: %f \n', eta);
                
                allResults = zeros(nreps,4);
                for v = 1:nreps
                    tic
                    [C, Y, obj_value, data_clustered] = cl_mg_v2(data, nbclusters, eta, {sigma, [k sigma], epsilon, m}, 'sym', algochoices, eigv); %***
                    toc
                    [clusters,~,~] = find(Y');  %change label matrix into column
                    clusterResults.MVMG = [clusterResults.MVMG, clusters]; 
                    
                    %evaluation
                    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                    MIhat = MutualInfo(label_ind, clusters);
                    disp(RI);
                    disp(MIhat);
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                    
                    %obtain the clusters results from highest performance
                    if mean(singleResult) > maxResult
                       maxResult = mean(singleResult);
                       clusterBestResults.MVMG.result = clusters;
                       clusterBestResults.MVMG.para.eta = eta;
                    end
                end
                result = mean(allResults,1); % result is average result;
                
                disp(['ACC, MIhat, Purity, F1score: ',num2str(result)]);
                fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                
            end
            
            
        case 'ONGC'
            
            clusterResults.ONGC = []; %initialize clusterResults.ONGC
            disp(method);
            fprintf(fid, [method,'\n']);
            
            if iscell(data)
                allData = cell2mat(data')';
            else
                allData = data';
            end
            
            nsample = size(allData,1);
            
            %% setting for ONGC !!!
            anchorCreateMethod = 'kmeans';
            maxResult = -inf; %predefine the varible to save the highest performance
            %             m = 300; for test other
            %             m = 300; for test MSCR_v1
            %             r = 2;
            %             k = 5;
            %             p = nbclusters;
            r = 2; %the decimation factor is set as 10 for all data sets 1 for Cal20_cnn, 5 for ApAy_cnn
            %except USPS which is set as 3 in ULGE paper. !!!
            k = 5; %default setting in ULGE paper
            p = nbclusters;
            
            %!!!
            spl_ratio = 0.04: 0.02: 0.2; %default !!!!
            % spl_ratio = 0.1: 0.02: 0.2; %default for msrcv1 !!!!
            % spl_ratio = 0.1; %for single test !!!
            nratio = numel(spl_ratio);
            iniMethod = 'orth_random'; % SPCL or random %initialisation method for ONGC
            paraMode = 'grid';
            
            if strcmp(paraMode,'grid')
                %mu_vec = [10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1, 10, 100, 1000, 10^4, 10^5 ]; %default !!!!
                mu_vec = [10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1, 10, 100]; %new default !!!!
                %mu_vec = [1];
            elseif strcmp(paraMode,'rand')
                a = -2;
                b = 2;     
                nmu = 20;
                mu_vec = 10.^((b-a).*rand(nmu,1) + a);
            end
            %% =========
            
            fprintf(fid, 'anchorCreateMethod: %s \n', anchorCreateMethod);
            fprintf(fid, 'r: %d \n', r);
            fprintf(fid, 'k: %d \n', k);
            fprintf(fid, 'p: %d \n\n', p);
            
            %% ======
            for j = 1:nratio
                
                m = round(spl_ratio(j)*nsample);
                
                fprintf(fid, 'm: %d \n', m);
                
                %!!!!
%                 % use ULGE graph
%                 L = ULGE(allData, anchorCreateMethod, m, r, k, p);
                
                % use normal gaussian graph
                sigma = determineSigma(allData', 1, 0.15);
                wmat = SimGraph_Full(allData', sigma);
                dmat = diag(sum(wmat, 2));
                L = eye(nsample) - (dmat^-0.5) * wmat * (dmat^-0.5);
                % L = (dmat^-0.5) * wmat * (dmat^-0.5);
                
                for t = 1:numel(mu_vec)
                    mu = mu_vec(t);
                    fprintf(fid, 'mu: %f \n', mu);
                    
                    for v = 1:nreps
                        
                        [clusters, F, oobj, mobj] = algONGC(L, nbclusters, mu, iniMethod);
                        % [clusters, F, oobj, mobj] = algONGC(L,round(nsample/2), mu, iniMethod);%for test
                        clusterResults.ONGC = [clusterResults.ONGC, clusters];
                        
                        %evaluation
                        singleResult = ClusteringMeasure(label_ind, clusters);
                        allResults(v,:) = singleResult;
                        
                        %obtain the clusters results from highest performance
                        if mean(singleResult) > maxResult
                            maxResult = mean(singleResult);
                            clusterBestResults.ONGC.result = clusters;
                            clusterBestResults.ONGC.para.m = m;
                            clusterBestResults.ONGC.para.mu = mu;
                            
                        end
                    end
                    result = mean(allResults,1); % result is average result;
                    SEM = std(allResults, 0, 1)/sqrt(length(nreps));
                    
                    disp(['ACC, MIhat, Purity, F1score: ',num2str(result),...
                        ' SEM: ',num2str(SEM),'\n\n']);
                    %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                    fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),...
                        ' SEM: ',num2str(SEM),'\n\n']);
                end
                
            end
    end
    
end

fclose(fid);

if exist('clusterBestResults','var')
    save(saveClusterResultsFile,'clusterResults','clusterBestResults'); % uncomment while saving
else
    save(saveClusterResultsFile,'clusterResults'); % uncomment while saving
end

catch ME
    fclose(fid);
    rethrow(ME);    
end

load gong.mat;
sound(y, 8*Fs);
