function [C_f, Y_f, obj_value, F_vf] = cl_mg_v2(data, nbclusters, eta, varargin)
       
%% Joint Multi-Graph Clustering algorithm
% --- details --- (option)
% cl_mg_v2(data, nbclusters, varargin) is the second version of clustering 
% algorithm via multi-graph joint learning. (JMGC)

% --- version ---- (option)
% cl_mg_v2  

% cl_mg

% --- Input ---
% data: 
%   the input data R^{d*n} d is dimension, n is number of samples
% nbclusters: 
%   the number of the clusters
% eta: 
%   the parameter of the algorithm
% varargin: 
%   optional parameters for initial spectral clustering
%     The first parameter of varargin is the name of the similarity
%       function to use. (if omitted the default will be using all functions)
%     The second is the parameters to pass to those similarity functions.
%     The third parameter is the type of the Laplacian matrix:
%       'unormalized' - unnormalized laplacian matrix
%       'sym' - normalized symmetric laplacian matrix
%       'rw' - normalized asymmetric laplacian matrix
%       (if omitted the default will be 'unnormalized')
%     The fourth parameter is the algorithm used for organizing eigenvectors:
%       'np' - generally used for 2 clusters, one eigenvector must be used, if
%        will put positive values in class 1 and negative values in class 2
%       'kmean' - a k-mean algorithm will be used to cluster the given eigenvectors
%     The fifth parameter is the eigenvector choice, manually set it as
%       eigv = [1 nbclusters];
%
% --- output ----
% C_f: nbclusters cluster centroid locations in R^{nbclusters * d} 
% Y_f: the indicating matrix {0,1}^{n * nbclusters}
% obj_value: the objective value of each iterations R^{1 * niterations}
% F_vf: the final clustered data in multi-graph. a cell with V elements each of which is R^{n * nbclusters}
% --- ref ---
% Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and computing 17.4 (2007): 395-416.
% 
% --- note ---(option)

% by Lance Liu 

%% parameter setting
niters = 100; % total number of interations
in_iters = 9; % number of increasing iters
% plotchoices = {'bo','r+','md','k*','wv'}; % use if visualize the results
lapmatrixchoices = {'unormalized', 'sym', 'rw'};
algochoices = {'np', 'kmeans'};
func = {'gaussdist','knn','eps_neighbor','CLR'};
    % = {'gaussdist','knn','eps_neighbor','CLR'};
V = numel(func); % number of graphs from data (may need to put this part into graph section later)
%eta = 0.01; %a fixed para, now has been seen as the input
alpha = ones(V, 1); %initial condition
%niters = 100;
%obj_value = zeros(1,niters); 

count = 1;

if iscell(data)
    data = data'; 
    X = cell2mat(data)'; % concatenated data
    % data: original split data type
    [~, ndata_vec] = cellfun(@size, data);
    n = ndata_vec(1);
else
    X = data;
    n = size(data,1);
end

%%get all the parameters%%%
if(ischar(varargin{count}))
        
    func = varargin{count};
    count = count + 1;
end

params = varargin{count};
count = count + 1;

if(length(varargin) >= count)
    
    if(sum(strcmp(varargin{count}, lapmatrixchoices)) == 0)

        lapmatrixchoice = 'unormalized';
    else

        lapmatrixchoice = varargin{count};
        count = count + 1;
    end

    if(length(varargin) >= count)
        
        if(sum(strcmp(varargin{count}, algochoices)) == 0)

            clusteralgo = 'np';
        else
            clusteralgo = varargin{count};
            count = count + 1;
        end

        if(length(varargin) >= count)

            eigv = varargin{count};
        else
            
            eigv = [2 2];  % eigv = [2 2 + nbclusters] 
        end
    else
        clusteralgo = 'np';
        eigv = [2 2];
    end
else
    
    lapmatrixchoice = 'unormalized';
    clusteralgo = 'np';
    eigv = [2 2];
end
%%all parameters are got%%%

sprintf('Graph choice is fully connected\nLaplacian choice is %s\nCluster algorithm is %s', lapmatrixchoice, clusteralgo)


%% Initial graph constructing
[nbsamples, dim] = size(X);
X = X'; %transpose as the introduction in this code

% initialization
for v = 1:V
    wmat = zeros(nbsamples);
    
    switch func{v}
        case 'gaussdist'
            wmat = SimGraph_Full(X, params{v});
            
        case 'knn'
            Type = 1; %Type = 1 normal, Type = 2 mutual
            k = params{v}(1); %number of neighborhood
            wmat = full(KnnGraph(X, k, Type, params{v}(2)));
            %[n,d]=knnsearch(x,y,'k',10,'distance','minkowski','p',5);
            
        case 'eps_neighbor'
            wmat = full(SimGraph_Epsilon(X, params{v}));
            
        case 'CLR'
            [~, wmat] = CLR_main(X, nbclusters, params{v});
            
        case 'chi_square' % problem here !!!
            for i = 1: nbsamples - 1
                wmat(i, i + 1: end) = feval(func{v}, repmat(X(i, :), nbsamples - i, 1), X(i + 1: end,:));
            end
            
    end

    %wmat = wmat + wmat';
    dmat = diag(sum(wmat, 2));

    switch lapmatrixchoice
        case 'unormalized'
            %A_norm{v} = dmat - wmat;
            A_norm{v} = wmat;
        case 'sym'
            %A_norm{v} = eye(nbsamples) - (dmat^-0.5) * wmat * (dmat^-0.5);
            A_norm{v} = (dmat^-0.5) * wmat * (dmat^-0.5);
        case 'rw'
            %A_norm{v} = eye(nbsamples) - (dmat^-1) * wmat;
            A_norm{v} = (dmat^-1) * wmat;
    end

    [U_,S,~] = svd(A_norm{v}, 'econ');
    
    % Unormalize on each row
    U = U_(:,  eigv(1,1)+1: eigv(1,2)+1);
    %F_v{v} = U;
    
    % Normalize each row to be of unit length
    
    sq_sum = sqrt(sum(U.*U, 2)) + 1e-20;
    F_v{v} = U ./ repmat(sq_sum, 1, nbclusters);
end

%???
% % Normalize each row to be of unit length
% sq_sum = sqrt(sum(V.*V, 2)) + 1e-20;
% U = V ./ repmat(sq_sum, 1, num_clusters);

% % online tuning
%     [U,S,~] = svd(A_norm{1}, 'econ');
%     %F_v{v} = U(:,  end-eigv(1,2): end-eigv(1,1));
%     %[U,~,V] = svd(M,'econ');
%     %F_v{v} = U(:,  end-eigv(1,2)+1: end);
%     F_v{1} = U(:,  eigv(1,1)+1: eigv(1,2)+1);
% %%%%
    
%% Optimization
tmp = 0;
[~, dim_V] = cellfun(@size, F_v');
for i = 1:V  % number of graphs from data (may need to put this part into graph section later)
    tmp = tmp + dim_V(i);
    dim_sum(i) = tmp;
end
dim_V_ind1 = [1,dim_sum(1:V-1)+1]; %start index of each view of features
dim_V_ind2 = [dim_sum(1:V-1),dim_sum(V)]; %end index of each view of features
    
max_rec = -inf; % the varible record the max objective value
count = 1; % counts of iters
in_count = 0; % counts of iters when objective value increase
while count <= niters
% fix F and alpha
    F_a = cell2mat(F_v);
    [idx, C] = kmeans(F_a, nbclusters); %change the dim of C into C_v (cluster according to each graph)
    
    Y = zeros(n, nbclusters);
    for j = 1:n
        Y(j,idx(j)) = 1;
    end
    
% fix Y and C   
    % ->fix F
       %B = Y*C;
    for v = 1:V
        B{v} = Y*C(:,dim_V_ind1(v):dim_V_ind2(v));
        nv(v) = trace(F_v{v}'*A_norm{v}*F_v{v});
    end
    %B = Y*C;
    
     lambda = norm(nv);
     alpha = nv./lambda;

    % ->fix alpha     
     for v = 1:V
        [F_v{v}, obj] = GPI(A_norm{v}, B{v}, eta/alpha(v));
        %problem here 1. how to set the third para for GPI, how to make the
        %B,use all feature to do kmeans cluster or sperate each feature.
     end
    
% calculate obj_value
    tmp = 0;
    for v = 1:V
        tmp = tmp + alpha(v)*trace(F_v{v}'*A_norm{v}*F_v{v});
    end
    obj_value(count) = tmp - eta * norm(cell2mat(F_v) - Y*C, 'fro')^2;
    
    if obj_value(count) > max_rec
        max_rec = obj_value(count);
        F_vf = F_v; % here _f means final out put value
        C_f = C;
        Y_f = Y;
        in_count = in_count + 1; %in_count means increasing count
    end
    
    if in_count > in_iters
       break; 
    end
    
    disp('iters:');
    disp(count);
    count = count + 1;
    
end

alpha; % stop here to show alpha
