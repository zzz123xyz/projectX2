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

%% ==== dataset and global para ==== %!!!!
%dataset_name = 'MSRCV1'; %'AWA','MSRCV1','NUSWIDEOBJ','Cal7','Cal20',
%'HW',AWA4000,'ApAy','AWA_MDR','ApAy_MDR'(rename from
%ApAy_MDR1,ApAy_MDR2... ),ApAy_MDR_R01R01R005, A
%'USAA','USAA_MDR_R005'
dataset_name = 'ApAy_4_trn'; 
featType = 'all'; % default : all
nreps = 1; % parameter  default : 1
clusterResults = struct;
% bestClusterPara = getBestPara(dataset_name); % using best parameters if
%there any !!!
saveClusterResultsFile = ['results\clusterResults_',dataset_name,'.mat'];

%% ==== selecting methods ==== %!!!!
%methods = {'kmeans'};
%methods = {'kmeans','SPCLNaive'};
%methods = {'kmeans','SPCL','SPCLNaive','MVSC','MMSC','MVCSS','MVG','CLR','MVMG'}; % default
%methods = {'SPCL'};
%methods = {'MVMG'};
%methods = {'MVG'};
%methods = {'CLR'};
%methods = {'kmeans','SPCL','SPCLNaive'};
%methods = {'MVG', 'CLR', 'MVMG'};
methods = {'SPCL', 'MVSC', 'CLR', 'MVG', 'MVMG'}; %chosen
%methods = {'kmeans', 'SPCL', 'MVSC', 'MMSC', 'MVCSS', 'MVG', 'CLR', 'MVMG'};


%% ==== read dataset ====
[data, label_ind] = readClusterDataset(dataset_name);
if ~strcmp(featType,'all')
    data = data{str2double(featType)};
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
            disp(method);
            fprintf(fid, [method,'\n']);
            if iscell(data)
                allData = cell2mat(data')';
            else
                allData = data;
            end
             
            allResults = zeros(nreps,3);
            for v = 1:nreps
                [clusters, center] = kmeans(allData, nbclusters);
                clusterResults.kmeans = clusters;
                
                %evaluation
                [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                MIhat = MutualInfo(label_ind, clusters);
                disp(RI);
                disp(MIhat);
                singleResult = ClusteringMeasure(label_ind, clusters);
                allResults(v,:) = singleResult;
            end
            result = mean(allResults,1); % result is average result;
            
            disp(['ACC, MIhat, Purity:',num2str(result)]);
            fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);
            
        case  'SPCL'
           %% spectral clustering
            disp(method);
            fprintf(fid, [method,'\n']);
            
            algochoices = 'kmean';
            fprintf(fid, 'algochoices: %s \n', algochoices);
            
            eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
            fprintf(fid, 'eigv: %s \n\n', num2str(eigv));
            
            if exist('bestClusterPara','var')
                pdpara = bestClusterPara.SPCL.sigma ; %the predefine sigma parameter 2.7 MSCRv1;
                %the predefine sigma parameter 4.32 Cal20;
            end
            
            for percent = 0.1: 0.05: 0.5; % the following if block can not used for this line, consider to change ***
                %percent = 0.05: 0.05: 0.5; % default !!!
                
                sigma = determineSigma(data, 1, percent);
                
                % if there is a predefine parameter
                if exist('pdpara','var')&&percent==0.05
                    sigma = pdpara;
                elseif  exist('pdpara','var')&&percent>0.05
                    clear pdpara;
                    break;
                end
                
                fprintf(fid, 'sigma: %f \n', sigma);
                
                allResults = zeros(nreps,3);
                for v = 1:nreps
                    [clusters, evalues, evectors] = spcl(data, nbclusters, sigma, 'sym', algochoices, eigv);
                    clusterResults.SPCL = clusters;
                    
                    %evaluation
                    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                    MIhat = MutualInfo(label_ind, clusters);
                    disp(RI);
                    disp(MIhat);
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                end
                result = mean(allResults,1); % result is average result;
                
                disp(['ACC, MIhat, Purity:',num2str(result)]);
                fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);
            end
            
        case 'SPCLNaive'
           %% spectral clustering Naive mode (apply the spectral clustering
            % algorithm on the combined Laplacian matrix which is
            % the summation of five Laplacian matrix corresponding
            % to each single modal)
            
            disp(method);
            fprintf(fid, [method,'\n']);
            
            func = 'gaussdist';
            fprintf(fid, 'fun: %s \n', func);
            
            algochoices = 'kmean';
            fprintf(fid, 'algochoices: %s \n', algochoices);
            
            eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
            fprintf(fid, 'eigv: %s \n', num2str(eigv));
            %sigma = 3000;

            %                 sigma = determineSigma(data, 1, percent);
            %                 fprintf(fid, 'sigma: %f \n', sigma);
            allResults = zeros(nreps,3);
            for v = 1:nreps
                [clusters, evalues, evectors] = spclNaive(data, nbclusters, func, 'sym', algochoices, eigv);
                clusterResults.SPCLNaive = clusters;
                
                %evaluation
                [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                MIhat = MutualInfo(label_ind, clusters);
                disp(RI);
                disp(MIhat);
                singleResult = ClusteringMeasure(label_ind, clusters);
                allResults(v,:) = singleResult;
            end
            result = mean(allResults,1); % result is average result;
                          
            disp(['ACC, MIhat, Purity:',num2str(result)]);
            fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);
            
        case 'MVSC'
            %% MVSC
            % Li, Y., Nie, F., Huang, H., & Huang, J. (2015, January).
            % Large-Scale Multi-View Spectral Clustering via Bipartite Graph.
            % In AAAI (pp. 2750-2756).
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
                
                allResults = zeros(nreps,3);
                for v = 1:nreps
                    
                    [clusters0, ~, obj_value, nbData] = MVSC(data, nbclusters, nbSltPnt, k, gamma, ...
                        func, sigma);
                    clusters = clusters0(1:nbData);
                    clusterResults.MVSC = clusters;
                    
                    %evaluation
                    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                    MIhat = MutualInfo(label_ind, clusters);
                    disp(RI);
                    disp(MIhat);
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                end
                result = mean(allResults,1); % result is average result;
                
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
                
                allResults = zeros(nreps,3);
                for v = 1:nreps
                    
                    %[clusters, obj_value] = MMSC(data, nbclusters, a, func, param);
                    Y = MMSC_main(data, nbclusters, a, func, param, discrete_model);
                    
                    if strcmp(discrete_model,'nmf')
                        clusters = kmeans(Y, nbclusters);
                    else
                        [clusters,~,~] = find(Y');  %change label matrix into column
                    end
                    
                    clusterResults.MMSC = clusters;
                    
                    %evaluation
                    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                    MIhat = MutualInfo(label_ind, clusters);
                    disp(RI);
                    disp(MIhat);
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                end
                result = mean(allResults,1); % result is average result;
                
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
                    
                    allResults = zeros(nreps,3);
                    for v = 1:nreps
                        [clusters, obj_value, F_record] = multi_view_fusion(data, nbclusters, gamma1, gamma2); % do pca on data first, No you can use pinv, right?
                        clusterResults.MVCSS = clusters;
                        
                        %evaluation
                        [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                        MIhat = MutualInfo(label_ind, clusters);
                        disp(RI);
                        disp(MIhat);
                        singleResult = ClusteringMeasure(label_ind, clusters);
                        allResults(v,:) = singleResult;
                    end
                    result = mean(allResults,1); % result is average result;
                    
                    disp(['ACC, MIhat, Purity:',num2str(result)]);
                    fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);
                end
            end
            
            
        case 'MVG'
            %% MVG
            % multi-view single graph joint clustering
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
                
                allResults = zeros(nreps,3);
                for v = 1:nreps
                    [C, Y, obj_value, data_clustered] = MVG(data, nbclusters, eta, eigv, 'CLR', m); %***
                    [clusters,~,~] = find(Y');  %change label matrix into column
                    clusterResults.MVG = clusters;
                    
                    %evaluation
                    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                    MIhat = MutualInfo(label_ind, clusters);
                    disp(RI);
                    disp(MIhat);
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                end
                result = mean(allResults,1); % result is average result;
                
                disp(['ACC, MIhat, Purity:',num2str(result)]);
                fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);
            end
            
            fprintf(fid, '\n');
            
        case 'CLR'
            %% CLR
            % Nie, Feiping, et al. "The Constrained Laplacian Rank Algorithm for Graph-Based Clustering." (2016).
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
                
                allResults = zeros(nreps,3);
                
                flag = 0;
                
                while flag == 0
                    try
                        for v = 1:nreps
                            [clusters, S, evectors, cs] = CLR_main(AllDataMatrix, nbclusters, m);
                            clusterResults.CLR = clusters;
                            
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
                disp(['ACC, MIhat, Purity:',num2str(result)]);
                fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);
            end
            fprintf(fid, '\n');
            
        case 'MVMG'
            %% MVMG
            % multi-graph joint spectral clustering
            %[C, obj_value, data_clustered] = cl_mg(data, nbclusters, {sigma, sigma, epsilon}, 'sym', 'kmean', [1 28]); %***
            disp(method);
            fprintf(fid, [method,'\n']);
            
            algochoices = 'kmean';
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
                
                allResults = zeros(nreps,3);
                for v = 1:nreps
                    tic
                    [C, Y, obj_value, data_clustered] = cl_mg_v2(data, nbclusters, eta, {sigma, [k sigma], epsilon, m}, 'sym', algochoices, eigv); %***
                    toc
                    [clusters,~,~] = find(Y');  %change label matrix into column
                    clusterResults.MVMG = clusters;
                    
                    %evaluation
                    [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                    MIhat = MutualInfo(label_ind, clusters);
                    disp(RI);
                    disp(MIhat);
                    singleResult = ClusteringMeasure(label_ind, clusters);
                    allResults(v,:) = singleResult;
                end
                result = mean(allResults,1); % result is average result;
                
                disp(['ACC, MIhat, Purity:',num2str(result)]);
                fprintf(fid, ['ACC, MIhat, Purity:',num2str(result),'\n\n']);
                
            end
    end
    
end

fclose(fid);
save(saveClusterResultsFile,'clusterResults'); % uncomment while saving

load gong.mat;
sound(y, 8*Fs);
