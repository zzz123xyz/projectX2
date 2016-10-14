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
addpath('..\clustering_eval_kun')

%dataset_name = 'MSRCV1'; %'AWA','MSRCV1','NUSWIDEOBJ','Cal7','Cal20',
%'HW',AWA4000
dataset_name = 'Cal20'; 
featType = 'all'; % default : all

%methods = {'kmeans'};
methods = {'kmeans','SPCLNaive'};
%methods = {'kmeans','SPCL','SPCLNaive', 'MVSC', 'MMSC', 'MVCSS', 'MVG', 'CLR', 'MVMG'};
%methods = {'kmeans','SPCL','SPCLNaive'};
%methods = {'kmeans', 'SPCL', 'CLR'};
%methods = {'kmeans', 'SPCL', 'MVSC', 'MVCSS', 'MVG', 'CLR', 'MVMG'};
%methods = {'kmeans', 'SPCL', 'MVSC', 'MMSC', 'MVCSS', 'MVG', 'CLR', 'MVMG'};

[data, label_ind] = readClusterDataset(dataset_name);
if ~strcmp(featType,'all')
    data = data{str2double(featType)};
end

nbclusters = numel(unique(label_ind));  %nbclusters =  7, 

fnPart = '';
for i = 1:numel(methods)
  fnPart = [fnPart,[methods{i},'_']];
end

name = dir('results/result_*');
k = numel(name);
ii = k+1;
OutputFile = ['results/result_',num2str(ii,'%03i'),'_',dataset_name,'_',featType,'_',num2str(nbclusters),'_',fnPart,'.txt'];
fid = fopen(OutputFile, 'wt');
fprintf(fid,['filename:',OutputFile,'\n']);
fprintf(fid,['dataset:',dataset_name,'\n']);
fprintf(fid,['number of clusters:',num2str(nbclusters),'\n']);
fprintf(fid,['methods:',fnPart,'\n\n']);

nmethod = numel(methods);

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
            fprintf(fid, 'algochoices: %s \n', algochoices);
            
            eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
            fprintf(fid, 'eigv: %s \n\n', num2str(eigv));
            %%pdsigma = 3.5; %the predefine sigma parameter 2.7;
            
            for percent = 0.05: 0.05: 0.5;
                
                sigma = determineSigma(data, 1, percent);
                % if there is a predefine sigma parameter
                if exist('pdsigma','var')&&percent==0.05
                    sigma = pdsigma;
                elseif  exist('pdsigma','var')&&percent>0.05
                    clear pdsigma;
                    break;
                end             
                
                fprintf(fid, 'sigma: %f \n', sigma);
                
                [clusters, evalues, evectors] = spcl(data, nbclusters, sigma, 'sym', algochoices, eigv);
                
                %evaluation
                [~,RI,~,~] = valid_RandIndex(label_ind, clusters);
                MIhat = MutualInfo(label_ind, clusters);
                disp(RI);
                disp(MIhat);
                result = ClusteringMeasure(label_ind, clusters);
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
            
            [clusters, evalues, evectors] = spclNaive(data, nbclusters, func, 'sym', algochoices, eigv);
            
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
            
            nbSltPnt = 400; %nbSltPnt = 40 for MSCRV1 %400  others
            sigma; % assume the value is as the same as in SPCL
            k = 8;
            func = 'gaussdist';
            fprintf(fid, 'nbSltPnt: %d \n', nbSltPnt);
            fprintf(fid, 'sigma: %f \n', sigma);
            fprintf(fid, 'k: %d \n', k);
            fprintf(fid, 'func: %s \n\n', func);
            param_list = 0.1:0.2:2;
            
            for j = 1:numel(param_list)
                t = param_list(j);
                gamma = 10^t; % gamma = 10; may need to be changed
                fprintf(fid, 'gamma: %f \n', gamma);
                
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
            %func = 'gaussdist';
            param = 10;  %in paper it's k=9 in P5, the first col is datapoints themselves
            %discrete_model = 'rotation';
            discrete_model = 'rotation';
            
            fprintf(fid, 'func: %s \n', func);
            fprintf(fid, 'param: %d \n', param);
            fprintf(fid, 'discrete_model: %s \n\n', discrete_model);
            for t = -2:0.2:2
                a = 10^t;
                fprintf(fid, 'a: %f \n', a);
                
                %[clusters, obj_value] = MMSC(data, nbclusters, a, func, param);
                Y = MMSC_main(data, nbclusters, a, func, param, discrete_model);
                
                if strcmp(discrete_model,'nmf')
                    clusters = kmeans(Y, nbclusters);
                else
                    [clusters,~,~] = find(Y');  %change label matrix into column
                end
                
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
            fprintf(fid, [method,'\n\n']);
            
            % exponent = [-5 : 5];
            exponent = 1;
            for i = 1: numel(exponent)
                gamma1 = 10^exponent(i);
                for j =  1: numel(exponent)
                    gamma2 = 10^exponent(j);
                    fprintf(fid, 'gamma1: %d \n', gamma1);
                    fprintf(fid, 'gamma2: %d \n', gamma2);
                    
                    [clusters, obj_value, F_record] = multi_view_fusion(data, nbclusters, gamma1, gamma2); % do pca on data first, No you can use pinv, right?
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
            
            
        case 'MVG'
            %% MVG
            % multi-view single graph joint clustering
            disp(method);
            fprintf(fid, [method,'\n']);
            
            m = 9;
            eigv = [1 nbclusters]; %eigv = [1 28], [2 2], [1 nbclusters];
            fprintf(fid, 'm: %d \n', m);
            fprintf(fid, 'eigv: %s \n\n', num2str(eigv));
            
            for t = -2:0.2:2
                
                eta = 10^t;
                fprintf(fid, 'eta: %f \n', eta);
                
                [C, Y, obj_value, data_clustered] = MVG(data, nbclusters, eta, eigv, 'CLR', m); %***
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
            
            fprintf(fid, '\n');
            
        case 'CLR'
            %% CLR
            % Nie, Feiping, et al. "The Constrained Laplacian Rank Algorithm for Graph-Based Clustering." (2016).
            disp(method);
            fprintf(fid, [method,'\n\n']);
            
            %m = 7; % a para to tune m <10 in paper
            
            for m = 2:10 % a para to tune m <10 in paper, right when perform on cal20
            %for m = 4:7 % a para to tune m <10 in paper, right when perform on cal20
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
            
            for t = -2:0.2:2
                eta = 10^t;
                fprintf(fid, 'eta: %f \n', eta);
                
                [C, Y, obj_value, data_clustered] = cl_mg_v2(data, nbclusters, eta, {sigma, [k sigma], epsilon, m}, 'sym', algochoices, eigv); %***
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
    
end

fclose(fid);

load gong.mat;
sound(y, 8*Fs);
