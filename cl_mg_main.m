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
%'USAA','USAA_MDR_R005','UPSP', Cal20_cnn, Cal20_cnn_MDR512, ApAy_cnn_MDR512;
% ApAy_trn_cnn_MDR512
dataset_name = 'Cal20';
featType = 'all'; % default : all
nreps = 5; % parameter  default : 1
clusterResults = struct;
%bestClusterPara = getBestPara(dataset_name); % using best parameters if
%there any, Use the chosen ones not always the best !!!
%saveClusterResultsFile = ['results\clusterResults_',dataset_name,'.mat'];

%% ==== selecting methods ==== %!!!!
%methods = {'kmeans'};
%methods = {'kmeans','SPCLNaive'};
%methods = {'kmeans','SPCL','SPCLNaive','MVSC','MMSC','MVCSS','MVG','CLR','MVMG'}; % default
%methods = {'SPCL'}; % for single view
%methods = {'MVMG'};
%methods = {'MVG'};
%methods = {'CLR'};
%methods = {'kmeans','SPCL','SPCLNaive'};
%methods = {'MVG', 'CLR', 'MVMG'};
%methods = {'SPCL', 'MVSC', 'MVG', 'CLR', 'MVMG'}; %chosen
%methods = {'MVCSS'};
%methods = {'ONGC'};
%methods = {'ONGC_SPCL'}; %ONGC with gaussian graph
%methods = {'ONGC_ULGE'};
%ONGC with ULGE graph (default for ONGC, if no appendix after '_', they are in this case)
%methods = {'newMethodTest'};
methods = {'MVG','MVMG'};
%methods = {'SPCL','MVSC','CLR'};

%% ==== read dataset ====
if ~isempty(strfind(dataset_name,'cnn'))
    [~, label_ind] = readClusterDataset(dataset_name);
    data = readDeepFeat(dataset_name, featType);
else
    [data, label_ind] = readClusterDataset(dataset_name);
    if ~strcmp(featType,'all')
        data = data{str2double(featType)};
    end
end

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
        if  strcmp(method,'kmeans')
            %% kmeans
            clusterResults.kmeans = []; %initialize clusterResults.kmeans
            clusterResults.kmeansmeasure = {}; %initialize clusterResults.MVMGmeasure
            % the varible to record measuring results ACC NMI etc.
            clusterResults.kmeansresult = {}; %initialize clusterResults.MVMGresult
            % the varible to record clustering results ACC NMI etc.
            
            disp(method);
            fprintf(fid, [method,'\n']);
            if iscell(data)
                allData = cell2mat(data')';
            else
                allData = data';
            end
            
            allResults = zeros(nreps,6);
            allReps = [];
            for v = 1:nreps
                [clusters, center] = kmeans(allData, nbclusters);
                clusterResults.kmeans = [clusterResults.kmeans, clusters];
                allReps = [allReps, clusters];
                
                %evaluation
                singleResult = ClusteringMeasure(label_ind, clusters);
                allResults(v,:) = singleResult;
                disp(num2str(singleResult))
            end
            
            result = mean(allResults,1); % result is average result;
            SEM = std(allResults, 0, 1)/sqrt(length(nreps));
            
            disp(['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                ' SEM: ',num2str(SEM),'\n\n']);
            %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
            fprintf(fid, ['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                ' SEM: ',num2str(SEM),'\n\n']);
            
            clusterResults.kmeansmeasure = [clusterResults.kmeansmeasure; {allResults}];
            clusterResults.kmeansresult = [clusterResults.kmeansresult; {allReps}];
            % record measuring results ACC NMI etc for all trials, the result of each
            % parameter setting is in one cell
            
        elseif  strcmp(method,'SPCL')
            %% spectral clustering
            clusterResults.SPCL = []; %initialize clusterResults.SPCL
            clusterResults.SPCLmeasure = {}; %initialize clusterResults.MVMGmeasure
            % the varible to record measuring results ACC NMI etc.
            clusterResults.SPCLresult = {}; %initialize clusterResults.MVMGresult
            % the varible to record clustering results ACC NMI etc.
            
            disp(method);
            fprintf(fid, [method,'\n']);
            
            %% spectral clustering setting !!!
            algochoices = 'kmeans';
            eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
            percent_vec = 0.05: 0.05: 0.5;
            %percent_vec = [[0.02: 0.01: 0.04],[0.5:0.1:0.8]]; %worse?
            %% ====
            
            fprintf(fid, 'algochoices: %s \n', algochoices);
            fprintf(fid, 'eigv: %s \n\n', num2str(eigv));
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.SPCL.sigma ; %the predefine sigma parameter 2.7 MSCRv1;
                %the predefine sigma parameter 4.32 Cal20;
            end
            
            for j = 1:numel(percent_vec)
                percent = percent_vec(j); % default !!!
                %percent = 0.1: 0.05: 0.5 % the following if block can
                %not be used for this line, consider to change ***
                
                sigma = determineSigma(data, 1, percent);
                
                % if there is a predefine parameter
                if exist('pdpara','var')&&percent==0.05
                    sigma = pdpara;
                elseif  exist('pdpara','var')&&percent>0.05
                    clear pdpara;
                    break;
                end
                
                fprintf(fid, 'sigma: %f \n', sigma);
                
                allResults = zeros(nreps,6);
                allReps = [];
                for v = 1:nreps
                    [clusters, evalues, evectors] = spcl(data, nbclusters, sigma, 'sym', algochoices, eigv);
                    clusterResults.SPCL = [clusterResults.SPCL, clusters];
                    allReps = [allReps, clusters];
                    
                    %evaluation
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                    disp(num2str(singleResult))
                end
                
                result = mean(allResults,1); % result is average result;
                SEM = std(allResults, 0, 1)/sqrt(length(nreps));
                
                disp(['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                fprintf(fid, ['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                
                clusterResults.SPCLmeasure = [clusterResults.SPCLmeasure; {allResults}];
                clusterResults.SPCLresult = [clusterResults.SPCLresult; {allReps}];
                % record measuring results ACC NMI etc for all trials, the result of each
                % parameter setting is in one cell
            end
            
        elseif  strcmp(method,'SPCLNaive')
            %% spectral clustering Naive mode (apply the spectral clustering
            % algorithm on the combined Laplacian matrix which is
            % the summation of five Laplacian matrix corresponding
            % to each single modal)
            
            clusterResults.SPCLNaive = []; %initialize clusterResults.SPCLNaive
            clusterResults.SPCLNaivemeasure = {}; %initialize clusterResults.MVMGmeasure
            % the varible to record measuring results ACC NMI etc.
            clusterResults.SPCLNaiveresult = {}; %initialize clusterResults.MVMGresult
            % the varible to record clustering results ACC NMI etc.
            
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
            
            allResults = zeros(nreps,6);
            allReps = [];
            for v = 1:nreps
                [clusters, evalues, evectors] = spclNaive(data, nbclusters, func, 'sym', algochoices, eigv);
                clusterResults.SPCLNaive = [clusterResults.SPCLNaive, clusters];
                allReps = [allReps, clusters];
                
                %evaluation
                singleResult = ClusteringMeasure(label_ind, clusters);
                allResults(v,:) = singleResult;
                disp(num2str(singleResult))
            end
            result = mean(allResults,1); % result is average result;
            SEM = std(allResults, 0, 1)/sqrt(length(nreps));
            
            disp(['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                ' SEM: ',num2str(SEM),'\n\n']);
            %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
            fprintf(fid, ['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                ' SEM: ',num2str(SEM),'\n\n']);
            
            clusterResults.SPCLNaivemeasure = [clusterResults.SPCLNaivemeasure; {allResults}];
            clusterResults.SPCLNaiveresult = [clusterResults.SPCLNaiveresult; {allReps}];
            % record measuring results ACC NMI etc for all trials, the result of each
            % parameter setting is in one cell
            
        elseif  strcmp(method,'MVSC')
            %% MVSC
            % Li, Y., Nie, F., Huang, H., & Huang, J. (2015, January).
            % Large-Scale Multi-View Spectral Clustering via Bipartite Graph.
            % In AAAI (pp. 2750-2756).
            
            clusterResults.MVSC = []; %initialize clusterResults.MVSC
            clusterResults.MVSCmeasure = {}; %initialize clusterResults.MVMGmeasure
            % the varible to record measuring results ACC NMI etc.
            clusterResults.MVSCresult = {}; %initialize clusterResults.MVMGresult
            % the varible to record clustering results ACC NMI etc.
            
            disp(method);
            fprintf(fid, [method,'\n']);
            
            %% MVSC setting !!!
            if exist('bestClusterPara','var')
                nbSltPnt = bestClusterPara.MVSC.nbSltPnt; %nbSltPnt = 40 for MSCRV1 %400  others
            else
                nbSltPnt = 400;
            end
            
            percent = 0.35;
            sigma = determineSigma(data, 1, percent); % assume the value is as the same as in SPCL
            k = 8;
            func = 'gaussdist';
            param_list = 0.1:0.2:2;
            %% ==============
            
            fprintf(fid, 'nbSltPnt: %d \n', nbSltPnt);
            fprintf(fid, 'sigma: %f \n', sigma);
            fprintf(fid, 'k: %d \n', k);
            fprintf(fid, 'func: %s \n\n', func);
            
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
                
                allResults = zeros(nreps,6);
                allReps = [];
                for v = 1:nreps
                    
                    [clusters0, ~, obj_value, nbData] = MVSC(data, nbclusters, nbSltPnt, k, gamma, ...
                        func, sigma);
                    clusters = clusters0(1:nbData);
                    clusterResults.MVSC = [clusterResults.MVSC, clusters];
                    allReps = [allReps, clusters];
                    
                    %evaluation
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                    disp(num2str(singleResult))
                end
                result = mean(allResults,1); % result is average result;
                SEM = std(allResults, 0, 1)/sqrt(length(nreps));
                
                disp(['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                fprintf(fid, ['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                
                clusterResults.MVSCmeasure = [clusterResults.MVSCmeasure; {allResults}];
                clusterResults.MVSCresult = [clusterResults.MVSCresult; {allReps}];
                % record measuring results ACC NMI etc for all trials, the result of each
                % parameter setting is in one cell
            end
            fprintf(fid, '\n');
            
        elseif  strcmp(method,'MMSC')
            %% MMSC
            %Cai, Xiao, et al. "Heterogeneous image feature integration via multi-modal
            %spectral clustering." Computer Vision and Pattern Recognition (CVPR),
            %2011 IEEE Conference on. IEEE, 2011.
            
            clusterResults.MMSC = []; %initialize clusterResults.MMSC
            clusterResults.MMSCmeasure = {}; %initialize clusterResults.MVMGmeasure
            % the varible to record measuring results ACC NMI etc.
            clusterResults.MMSCresult = {}; %initialize clusterResults.MVMGresult
            % the varible to record clustering results ACC NMI etc.
            
            disp(method);
            fprintf(fid, [method,'\n']);
            
            %% MMSC setting !!!
            func = 'SelfTune';
            %func = 'gaussdist';
            param = 10;  %in paper it's k=9 in P5, the first col is datapoints themselves
            %discrete_model = 'rotation';
            discrete_model = 'rotation';
            t_vec = [-2:0.2:2];
            %% =====
            
            fprintf(fid, 'func: %s \n', func);
            fprintf(fid, 'param: %d \n', param);
            fprintf(fid, 'discrete_model: %s \n\n', discrete_model);
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.MMSC.a;
            end
            
            for j = 1:numel(t_vec)
                a = 10^t_vec(j);
                
                % if there is a predefine parameter
                if exist('pdpara','var')&&j==1
                    a = pdpara;
                elseif  exist('pdpara','var')&&j>1
                    clear pdpara;
                    break;
                end
                
                fprintf(fid, 'a: %f \n', a);
                
                allResults = zeros(nreps,6);
                allReps = [];
                for v = 1:nreps
                    
                    %[clusters, obj_value] = MMSC(data, nbclusters, a, func, param);
                    Y = MMSC_main(data, nbclusters, a, func, param, discrete_model);
                    
                    if strcmp(discrete_model,'nmf')
                        clusters = kmeans(Y, nbclusters);
                    else
                        [clusters,~,~] = find(Y');  %change label matrix into column
                    end
                    
                    clusterResults.MMSC = [clusterResults.MMSC, clusters];
                    allReps = [allReps, clusters];
                    
                    %evaluation
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                    disp(num2str(singleResult))
                end
                result = mean(allResults,1); % result is average result;
                SEM = std(allResults, 0, 1)/sqrt(length(nreps));
                
                disp(['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                fprintf(fid, ['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                
                clusterResults.MMSCmeasure = [clusterResults.MMSCmeasure; {allResults}];
                clusterResults.MMSCresult = [clusterResults.MMSCresult; {allReps}];
                % record measuring results ACC NMI etc for all trials, the result of each
                % parameter setting is in one cell
            end
            fprintf(fid, '\n');
            
        elseif  strcmp(method,'MVCSS')
            %% MVCSS
            % Multi-View Clustering and Feature Learning via Structured Sparsity
            % Wang, Hua, Feiping Nie, and Heng Huang. "Multi-view clustering and feature learning via structured sparsity."
            % Proceedings of the 30th International Conference on Machine Learning (ICML-13). 2013.
            tic
            
            clusterResults.MVCSS = []; %initialize clusterResults.MVCSS
            clusterResults.MVCSSmeasure = {}; %initialize clusterResults.MVMGmeasure
            % the varible to record measuring results ACC NMI etc.
            clusterResults.MVCSSresult = {}; %initialize clusterResults.MVMGresult
            % the varible to record clustering results ACC NMI etc.
            
            disp(method);
            fprintf(fid, [method,'\n\n']);
            
            %% MVCSS setting !!!
            %             exponent = [-5 : 5]; % gamm1 gamma2 in P6 in paper
            exponent1 = [-4:1]; %the best para from tab_selected_para.xlsx
            exponent2 = [-5:1]; %the best para from tab_selected_para.xlsx
            %% ====
            
            if exist('bestClusterPara','var')
                pdpara1 = bestClusterPara.MVCSS.gamma1;
                pdpara2 = bestClusterPara.MVCSS.gamma2;
            end
            
            for k = 1: numel(exponent1)
                gamma1 = 10^exponent1(k);
                
                % if there is a predefine parameter
                if exist('pdpara1','var')&&k==1
                    gamma1 = pdpara1;
                elseif  exist('pdpara1','var')&&k>1
                    clear pdpara1;
                    break;
                end
                
                for j =  1: numel(exponent2)
                    gamma2 = 10^exponent2(j);
                    
                    % if there is a predefine parameter
                    if exist('pdpara2','var')&&j==1
                        gamma2 = pdpara2;
                    elseif  exist('pdpara2','var')&&j>1
                        clear pdpara2;
                        break;
                    end
                    
                    fprintf(fid, 'gamma1: %d \n', gamma1);
                    fprintf(fid, 'gamma2: %d \n', gamma2);
                    
                    allResults = zeros(nreps,6);
                    allReps = [];
                    for v = 1:nreps
                        [clusters, obj_value, F_record] = multi_view_fusion(data, nbclusters, gamma1, gamma2); % do pca on data first, No you can use pinv, right?
                        clusterResults.MVCSS = [clusterResults.MVCSS, clusters];
                        allReps = [allReps, clusters];
                        
                        %evaluation
                        singleResult = ClusteringMeasure(label_ind, clusters);
                        allResults(v,:) = singleResult;
                        disp(num2str(singleResult))
                    end
                    result = mean(allResults,1); % result is average result;
                    SEM = std(allResults, 0, 1)/sqrt(length(nreps));
                    
                    disp(['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                        ' SEM: ',num2str(SEM),'\n\n']);
                    %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                    fprintf(fid, ['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                        ' SEM: ',num2str(SEM),'\n\n']);
                    
                    clusterResults.MVCSSmeasure = [clusterResults.MVCSSmeasure; {allResults}];
                    clusterResults.MVCSSresult = [clusterResults.MVCSSresult; {allReps}];
                    % record measuring results ACC NMI etc for all trials, the result of each
                    % parameter setting is in one cell
                    
                end
            end
            fprintf(fid, '\n');
            
            toc
        elseif  strcmp(method,'MVG')
            %% MVG
            % multi-view single graph joint clustering
            
            clusterResults.MVG = []; %initialize clusterResults.MVG
            clusterResults.MVGmeasure = {}; %initialize clusterResults.MVMGmeasure
            % the varible to record measuring results ACC NMI etc.
            clusterResults.MVGresult = {}; %initialize clusterResults.MVMGresult
            % the varible to record clustering results ACC NMI etc.
            
            disp(method);
            fprintf(fid, [method,'\n']);
            
            %% MVG setting !!!
            m = 9; % the best setting from experiment in CLR
            if strncmpi(dataset_name,'ApAy_MDR',8)
                %compare first 8 char to determine if the dataset is ApAy_MDR
                m = [9, 9, 30];
            elseif  strncmpi(dataset_name,'USAA',4)
                %compare first 4 char to determine if the dataset is USAA
                m = 8;
            end
            
            eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
            %% =====================
            
            fprintf(fid, 'm: %d \n', m);
            fprintf(fid, 'eigv: %s \n\n', num2str(eigv));
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.MVG.eta;
            end
            
            maxResult = -inf; %predefine the varible to save the highest performance
            for t = -2:0.2:2 % another setting place !!!
                eta = 10^t;
                
                % if there is a predefine parameter
                if exist('pdpara','var')&&t==-2
                    eta = pdpara;
                elseif  exist('pdpara','var')&&t>-2
                    clear pdpara;
                    break;
                end
                
                fprintf(fid, 'eta: %f \n', eta);
                
                allResults = zeros(nreps,6);
                allReps = [];
                for v = 1:nreps
                    [C, Y, obj_value, data_clustered] = MVG(data, nbclusters, eta, eigv, 'CLR', m); %***
                    [clusters,~,~] = find(Y');  %change label matrix into column
                    clusterResults.MVG = [clusterResults.MVG, clusters];
                    allReps = [allReps, clusters];
                    
                    %evaluation
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                    disp(num2str(singleResult))
                    
                    %obtain the clusters results from highest performance
                    if mean(singleResult) > maxResult
                        maxResult = mean(singleResult);
                        clusterBestResults.MVG.result = clusters;
                        clusterBestResults.MVG.para.eta = eta;
                    end
                end
                result = mean(allResults,1); % result is average result;
                SEM = std(allResults, 0, 1)/sqrt(length(nreps));
                
                disp(['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                fprintf(fid, ['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                
                clusterResults.MVGmeasure = [clusterResults.MVGmeasure; {allResults}];
                clusterResults.MVGresult = [clusterResults.MVGresult; {allReps}];
                % record measuring results ACC NMI etc for all trials, the result of each
                % parameter setting is in one cell
            end
            
            fprintf(fid, '\n');
            
        elseif  strcmp(method, 'CLR')
            %% CLR
            % Nie, Feiping, et al. "The Constrained Laplacian Rank Algorithm for Graph-Based Clustering." (2016).
            
            clusterResults.CLR = []; %initialize clusterResults.CLR
            clusterResults.CLRmeasure = {}; %initialize clusterResults.MVMGmeasure
            % the varible to record measuring results ACC NMI etc.
            clusterResults.CLRresult = {}; %initialize clusterResults.MVMGresult
            % the varible to record clustering results ACC NMI etc.
            
            disp(method);
            fprintf(fid, [method,'\n\n']);
            
            AllDataMatrix = DataConcatenate(data);
            
            %% CLR setting ==============
            %m = 7; % a para to tune m <10 in paper
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.CLR.m;
            end
            
            for m = 2:10 % a para to tune m <10 in paper, right when perform on cal20
                %for m = 4:7 % a para to tune m <10 in paper, right when perform on cal20
                %%  ==========
                % if there is a predefine parameter
                if exist('pdpara','var')&&m==2
                    m = pdpara;
                elseif  exist('pdpara','var')&&m>2
                    clear pdpara;
                    break;
                end
                
                allResults = zeros(nreps,6);
                allReps = [];
                
                flag = 0;
                while flag == 0
                    try
                        for v = 1:nreps
                            [clusters, S, evectors, cs] = CLR_main(AllDataMatrix, nbclusters, m);
                            clusterResults.CLR = [clusterResults.CLR, clusters];
                            allReps = [allReps, clusters];
                            
                            %evaluation
                            singleResult = ClusteringMeasure(label_ind, clusters);
                            allResults(v,:) = singleResult;
                            disp(num2str(singleResult))
                            
                            flag = 1;
                        end
                    catch
                        warning('Problem: set new m value because nbclusters is less than number of connected components');
                        m = m + 1;
                    end
                    
                end
                
                result = mean(allResults,1); % result is average result;
                SEM = std(allResults, 0, 1)/sqrt(length(nreps));
                
                fprintf(fid, 'm: %d \n', m);
                disp(['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                fprintf(fid, ['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                
                clusterResults.CLRmeasure = [clusterResults.CLRmeasure; {allResults}];
                clusterResults.CLRresult = [clusterResults.CLRresult; {allReps}];
                % record measuring results ACC NMI etc for all trials, the result of each
                % parameter setting is in one cell
            end
            fprintf(fid, '\n');
            
        elseif  strcmp(method, 'MVMG')
            %% MVMG
            % multi-graph joint spectral clustering
            %[C, obj_value, data_clustered] = cl_mg(data, nbclusters, {sigma, sigma, epsilon}, 'sym', 'kmeans', [1 28]); %***
            
            clusterResults.MVMG = []; %initialize clusterResults.MVMG
            clusterResults.MVMGmeasure = {}; %initialize clusterResults.MVMGmeasure
            % the varible to record measuring results ACC NMI etc.
            clusterResults.MVMGresult = {}; %initialize clusterResults.MVMGresult
            % the varible to record clustering results ACC NMI etc.
            
            disp(method);
            fprintf(fid, [method,'\n']);
            
            %% MVMG setting !!!! ============
            algochoices = 'kmeans';
            %sigma = 3000;
            m = 9; % the best setting from experiment in CLR
            sigma = determineSigma(data, 1, 0.15);
            epsilon = sigma;
            k = 20;
            eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
            %% ===============
            
            fprintf(fid, 'algochoices: %s \n', algochoices);
            fprintf(fid, 'sigma: %f \n', sigma);
            fprintf(fid, 'epsilon: %f \n', epsilon);
            fprintf(fid, 'k: %d \n', k);
            fprintf(fid, 'eigv: %s \n\n', num2str(eigv));
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.MVMG.eta;
            end
            
            maxResult = -inf; %predefine the varible to save the highest performance
            for t = -2:0.2:2 % another setting place !!!
                eta = 10^t;
                
                % if there is a predefine parameter
                if exist('pdpara','var')&&t==-2
                    eta = pdpara;
                elseif  exist('pdpara','var')&&t>-2
                    clear pdpara;
                    break;
                end
                
                fprintf(fid, 'eta: %f \n', eta);
                
                allResults = zeros(nreps,6);
                allReps = [];
                for v = 1:nreps
                    tic
                    [C, Y, obj_value, data_clustered] = cl_mg_v2(data, nbclusters, eta, {sigma, [k sigma], epsilon, m}, 'sym', algochoices, eigv); %***
                    toc
                    [clusters,~,~] = find(Y');  %change label matrix into column
                    clusterResults.MVMG = [clusterResults.MVMG, clusters];
                    allReps = [allReps, clusters];
                    
                    %evaluation
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                    disp(num2str(singleResult))
                    
                    %obtain the clusters results from highest performance
                    if mean(singleResult) > maxResult
                        maxResult = mean(singleResult);
                        clusterBestResults.MVMG.result = clusters;
                        clusterBestResults.MVMG.para.eta = eta;
                    end
                end
                
                result = mean(allResults,1); % result is average result;
                SEM = std(allResults, 0, 1)/sqrt(length(nreps));
                
                disp(['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                fprintf(fid, ['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                
                clusterResults.MVMGmeasure = [clusterResults.MVMGmeasure; {allResults}];
                clusterResults.MVMGresult = [clusterResults.MVMGresult; {allReps}];
                % record measuring results ACC NMI etc for all trials, the result of each
                % parameter setting is in one cell
            end
            
            
        elseif  ~isempty(strfind(method,'ONGC'))
            
            clusterResults.ONGC = []; %initialize clusterResults.ONGC
            clusterResults.ONGCmeasure = {}; %initialize clusterResults.ONGCmeasure
            % the varible to record measuring results ACC NMI etc.
            clusterResults.ONGCresult = {}; %initialize clusterResults.ONGCresult
            % the varible to record clustering results ACC NMI etc.
            
            
            disp(method);
            fprintf(fid, [method,'\n']);
            
            if iscell(data)
                allData = cell2mat(data')';
            else
                allData = data'; % allData = data'; for cnn feature?
            end
            
            nsample = size(allData,1);
            
            %% setting for ONGC !!
            anchorCreateMethod = 'kmeans';
            maxResult = -inf; %predefine the varible to save the highest performance
            %             m = 300; for test other
            %             m = 300; for test MSCR_v1
            %             r = 2;
            %             k = 5;
            %             p = nbclusters;
            r = 2; %the decimation factor is set as 10 for all data sets 1 for Cal20_cnn, 5 for ApAy_cnn
            %except USPS which is set as 3 in ULGE paper. 2 for 6000-8000 samples
            k = 5; %default setting in ULGE paper
            p = nbclusters;
            %!!!
            %spl_ratio = 0.04: 0.02: 0.2; %default !!!!
            %spl_ratio = 0.1: 0.02: 0.2; %default for msrcv1 !!!!
            %spl_ratio = 0.2: 0.02: 0.3; %test for the rest paras
            spl_ratio = 0.04; %fog single test
            nratio = numel(spl_ratio);
            iniMethod = 'orth_random'; % SPCL or random %initialisation method for ONGC
            paraMode = 'grid';
            
            if strcmp(paraMode,'grid')
                %mu_vec = [10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1, 10, 100, 1000, 10^4, 10^5 ]; %default !!!! for HW??
                %mu_vec = [10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1, 10, 100]; %new default !!!!
                mu_vec = [1000, 10^4, 10^5, 10^6, 10^7, 10^8]; %test for the rest paras
                %mu_vec = [10];
            elseif strcmp(paraMode,'rand')
                a = -2;
                b = 2;
                nmu = 20;
                mu_vec = 10.^((b-a).*rand(nmu,1) + a);
            end
            %% ======
            
            fprintf(fid, 'anchorCreateMethod: %s \n', anchorCreateMethod);
            fprintf(fid, 'r: %d \n', r);
            fprintf(fid, 'k: %d \n', k);
            fprintf(fid, 'p: %d \n\n', p);
            
            for j = 1:nratio
                
                %!!!!
                if ~isempty(strfind(method,'ULGE'))
                    % use ULGE graph
                    m = round(spl_ratio(j)*nsample);
                    [~, L] = ULGE(allData, anchorCreateMethod, m, r, k, p);
                    fprintf(fid, 'm: %d \n', m);
                    
                elseif ~isempty(strfind(method,'SPCL'))
                    % use normal gaussian graph
                    sigma = determineSigma(allData', 1, spl_ratio(j)); %the 3rd imput used to be 0.15 only(if not in the result txt)
                    wmat = SimGraph_Full(allData', sigma);
                    dmat = diag(sum(wmat, 2));
                    % L = (dmat^-0.5) * wmat * (dmat^-0.5);
                    L = eye(nsample) - (dmat^-0.5) * wmat * (dmat^-0.5);
                    fprintf(fid, 'sigma: %d \n', sigma);
                    
                end
                
                for t = 1:numel(mu_vec)
                    mu = mu_vec(t);
                    fprintf(fid, 'mu: %f \n', mu);
                    
                    allResults = zeros(nreps,6);
                    allReps = [];
                    for v = 1:nreps
                        [clusters, F, oobj, mobj] = algONGC(L, nbclusters, mu, iniMethod);
                        % [clusters, F, oobj, mobj] = algONGC(L,round(nsample/2), mu, iniMethod);%for test
                        clusterResults.ONGC = [clusterResults.ONGC, clusters];
                        allReps = [allReps, clusters];
                        
                        %evaluation
                        singleResult = ClusteringMeasure(label_ind, clusters);
                        allResults(v,:) = singleResult;
                        disp(num2str(singleResult))
                        
                        %obtain the clusters results from highest performance
                        if mean(singleResult) > maxResult
                            maxResult = mean(singleResult);
                            clusterBestResults.ONGC.result = clusters;
                            if exist('m','var')
                                clusterBestResults.ONGC.para.m = m;
                            elseif exist('sigma','var')
                                clusterBestResults.ONGC.para.sigma = sigma;
                            end
                            clusterBestResults.ONGC.para.mu = mu;
                        end
                    end
                    result = mean(allResults,1); % result is average result;
                    SEM = std(allResults, 0, 1)/sqrt(length(nreps));
                    
                    disp(['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                        ' SEM: ',num2str(SEM),'\n\n']);
                    %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                    fprintf(fid, ['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                        ' SEM: ',num2str(SEM),'\n\n']);
                    
                    clusterResults.ONGCmeasure = [clusterResults.ONGCmeasure; {allResults}];
                    clusterResults.ONGCresult = [clusterResults.ONGCresult; {allReps}];
                    % record measuring results ACC NMI etc for all trials, the result of each
                    % parameter setting is in one cell
                end
                
            end
            
        elseif  strcmp(method, 'newMethodTest')
            % newMethodTest is the combination of graph created by ULGC and
            % the triditional clustering method SPCL
            clusterResults.newMethodTest = []; %initialize clusterResults.newMethodTest
            clusterResults.newMethodTestmeasure = {}; %initialize clusterResults.newMethodTestmeasure
            % the varible to record measuring results ACC NMI etc.
            clusterResults.newMethodTestresult = {}; %initialize clusterResults.newMethodTestresult
            % the varible to record clustering results ACC NMI etc.
            
            
            if iscell(data)
                allData = cell2mat(data')';
            else
                allData = data';
            end
            
            %% setting for ULGC + SPCL!!
            spl_ratio = 0.04: 0.02: 0.2; %default !!!!
            %spl_ratio = 0.1: 0.02: 0.2; %default for msrcv1 !!!!
            %spl_ratio = 0.1;
            nratio = numel(spl_ratio);
            
            nsample = size(allData,1);
            m = round(spl_ratio*nsample);
            r = 1; %the decimation factor is set as 10 for all data sets
            %except USPS which is set as 3 in ULGE paper.
            k = 5; %default setting in ULGE paper
            p = nbclusters;
            eigv = [1 nbclusters];
            anchorCreateMethod = 'kmeans';
            clusteralgo = 'kmeans';
            %% ======
            
            maxResult = -inf;   %predefine the varible to save the highest performance
            for j = 1:nratio
                
                m = round(spl_ratio(j)*nsample);
                fprintf(fid, 'm: %d \n', m);
                
                allResults = zeros(nreps,6);
                allReps = [];
                for v = 1:nreps
                    [~, L] = ULGE(allData, anchorCreateMethod, m, r, k, p);
                    L = (L+L')/2; % make the constructed graph symmetric to avoid small
                    %computational turbulence which cause symmetric elements
                    
                    [evectors, evalues] = eigs(L, eigv(1,2)+1, 'sm');
                    newspace = evectors(:,2:end);
                    
                    % Normalize each row to be of unit length
                    sq_sum = sqrt(sum(newspace.*newspace, 2)) + 1e-20;
                    newspace = newspace ./ repmat(sq_sum, 1, nbclusters);
                    clear sq_sum;
                    
                    if(strcmp(clusteralgo, 'kmeans'))
                        
                        clusters = kmeans(newspace, nbclusters);
                    else
                        clusters = 1 + (newspace > 0);
                    end
                    
                    clusterResults.newMethodTest = [clusterResults.newMethodTest, clusters];
                    allReps = [allReps, clusters];
                    
                    %evaluation
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                    disp(num2str(singleResult))
                    
                    %obtain the clusters results from highest performance
                    if mean(singleResult) > maxResult
                        maxResult = mean(singleResult);
                        clusterBestResults.newMethodTest.result = clusters;
                        clusterBestResults.newMethodTest.para.m = m;
                    end
                end
                result = mean(allResults,1); % result is average result;
                
                result = mean(allResults,1); % result is average result;
                SEM = std(allResults, 0, 1)/sqrt(length(nreps));
                
                disp(['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                %fprintf(fid, ['ACC, MIhat, Purity, F1score: ',num2str(result),'\n\n']);
                fprintf(fid, ['ACC, MIhat, Purity, F1score, RI, Jaccard: ',num2str(result),...
                    ' SEM: ',num2str(SEM),'\n\n']);
                
                clusterResults.newMethodTestmeasure = [clusterResults.newMethodTestmeasure; {allResults}];
                clusterResults.newMethodTestresult = [clusterResults.newMethodTestresult; {allReps}];
                % record measuring results ACC NMI etc for all trials, the result of each
                % parameter setting is in one cell
                
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
    if exist('clusterBestResults','var')
        save(saveClusterResultsFile,'clusterResults','clusterBestResults'); % uncomment while saving
    else
        save(saveClusterResultsFile,'clusterResults'); % uncomment while saving
    end
    
    rethrow(ME);
end

load gong.mat;
sound(y, 8*Fs);
