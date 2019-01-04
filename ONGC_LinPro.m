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
dataset_name = 'ApAy_trn_cnn_MDR512';
featType = '3'; % default : all
nreps = 1; % parameter  default : 1
clusterResults = struct;
%bestClusterPara = getBestPara(dataset_name); % using best parameters if
%there any, Use the chosen ones not always the best !!!
%saveClusterResultsFile = ['results\clusterResults_',dataset_name,'.mat'];

%% ==== selecting methods ==== %!!!!
method = 'ONGC_LinPro_SPCL'; %ONGC with gaussian graph
%method = {'ONGC_LinPro_ULGE'};
%ONGC with ULGE graph (default for ONGC, if no appendix after '_', they are in this case)

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

%% ==== name the save files ====

name = dir('results/result_*');
k = numel(name);
ii = k+1;
OutputFile = ['results/result_',num2str(ii,'%03i'),'_',dataset_name,'_',featType,'_',num2str(nbclusters),'_',method,'_ave_',num2str(nreps),'.txt'];
saveClusterResultsFile = ['results/result_',num2str(ii,'%03i'),'_',dataset_name,'_',featType,'_',num2str(nbclusters),'_',method,'_ave_',num2str(nreps),'.mat'];

try
    fid = fopen(OutputFile, 'wt');
    
    fprintf(fid,['filename:',OutputFile,'\n']);
    fprintf(fid,['dataset:',dataset_name,'\n']);
    fprintf(fid,['number of clusters:',num2str(nbclusters),'\n']);
    fprintf(fid,['number of repeats:',num2str(nreps),'\n']);
    fprintf(fid,['methods:',method,'\n\n']);
    
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
%         mu_vec = [10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1, 10, 100, 1000, 10^4, 10^5, 10^6, 10^7, 10^8]; %default
%         gamma_vec = [10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1, 10, 100, 1000, 10^4, 10^5, 10^6, 10^7, 10^8]; %default
%         etag_vec = [10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1, 10, 100, 1000, 10^4, 10^5, 10^6, 10^7, 10^8]; %default
        mu_vec = [10^-2, 10^-1, 1, 10, 100, 1000, 10^4, 10^5]; %for efficiency
        gamma_vec = [10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1, 10, 100, 1000, 10^4, 10^5]; %for efficiency
        etag_vec = [10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1, 10, 100, 1000, 10^4, 10^5]; %for efficiency
        %mu_vec = [10];
    elseif strcmp(paraMode,'rand') % under construction***
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
        
        for t1 = 1:numel(mu_vec)
            para.mu = mu_vec(t1);
            
            for t2 = 1:numel(gamma_vec)
                para.gamma = gamma_vec(t2);
                
                for t3 = 1:numel(etag_vec)
                    para.etag = etag_vec(t3);
                    
                    fprintf(fid, 'mu: %f \n', para.mu);
                    fprintf(fid, 'gamma: %f \n', para.gamma);
                    fprintf(fid, 'etag: %f \n', para.etag);
                    
                    allResults = zeros(nreps,6);
                    allReps = [];
                    try
                        for v = 1:nreps
                            [clusters, F, oobj, mobj] = algONGC_LinPro(L, allData', nbclusters, para, iniMethod);
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
                                clusterBestResults.ONGC.para.mu = para.mu;
                            end
                        end
                    catch
                        warning('KMEANS does not accept complex data, the current parameters cannot be used, now continue with next parameteers');
                        continue;
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
