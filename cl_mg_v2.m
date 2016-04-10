function [C, Y, obj_value, data] = cl_mg_v2(data, nbclusters, varargin)
%
% spcl(data, nbclusters, varargin) is a spectral clustering function to
% assemble random unknown data into clusters. after specifying the data and
% the number of clusters, next parameters can vary as wanted. This function
% will construct the fully connected similarity graph of the data.
%
% The first parameter of varargin is the name of the function to use.
%
% The second is the parameter to pass to the function.

% Third parameter is the type of the Laplacian matrix:
% 'unormalized' - unnormalized laplacian matrix
% 'sym' - normalized symmetric laplacian matrix
% 'rw' - normalized asymmetric laplacian matrix
% (if omitted the default will be 'unnormalized')
% 
% then the algorithm used for organizing eigenvectors:
% 'np' - generally used for 2 clusters, one eigenvector must be used, if
% will put positive values in class 1 and negative values in class 2
% 'kmean' - a k-mean algorithm will be used to cluster the given eigenvectors
% 
% finally an eigenvector choice can be added, it can be a vector [vmin
% vmax] or a matrix defining several intervals. if not found the default
% will be [2 2]

%% -----------------------

% wmat \in R^{N*N} A in draft
% dmat \in R^{N*N} D in draft

%c is the number of clusters, d is dim of feature, n is number of samples
% F \in R^{d*n}
% Y \in R^{d*k}
% C \in {0,1}^{n*k}
%% ----------------------
        
%[d,n] = size(data);          
data = data'; 

plotchoices = {'bo','r+','md','k*','wv'};
lapmatrixchoices = {'unormalized', 'sym', 'rw'};
algochoices = {'np', 'kmean'};
func = {'gaussdist','knn','eps_neighbor'};
V = numel(func); % number of graphs from data (may need to put this part into graph section later)
eta = 1; %a fixed para
alpha = ones(V, 1); %initial condition
%niters = 100;
%obj_value = zeros(1,niters); 

count = 1;

if iscell(data)
    X = cell2mat(data)'; % concatenated data
    % data: original split data type
    [~, n_vec] = cellfun(@size, data);
    n = n_vec(1);
else
    X = data;
    n = size(data,1);
end

% 
% tmp = 0;
% for i = 1:V  % number of graphs from data (may need to put this part into graph section later)
%     tmp = tmp + dim_V(i);
%     dim_sum(i) = tmp;
% end
% dim_V_ind1 = [1,dim_sum(1:V-1)+1];
% dim_V_ind2 = [dim_sum(1:V-1),dim_sum(V)];

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

[nbsamples, dim] = size(X);
X = X'; %transpose as the intro in this code

% initialization
for v = 1:V
    wmat = zeros(nbsamples);
    
    switch func{v}
        case 'gaussdist'
            wmat = SimGraph_Full(X, params{v});
            
        case 'knn'
            Type = 1; %Type = 1 normal, Type = 2 mutual
            k = 10; %number of neighborhood
            wmat = full(KnnGraph(X, k, Type, params{v}));
            %[n,d]=knnsearch(x,y,'k',10,'distance','minkowski','p',5);
            
        case 'eps_neighbor'
            wmat = full(SimGraph_Epsilon(X, params{v}));
            
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

    [U,S,~] = svd(A_norm{v}, 'econ');
    %F_v{v} = U(:,  end-eigv(1,2): end-eigv(1,1));
    %[U,~,V] = svd(M,'econ');
    %F_v{v} = U(:,  end-eigv(1,2)+1: end);
    F_v{v} = U(:,  eigv(1,1): eigv(1,2));
end

tmp = 0;
[~, dim_V] = cellfun(@size, F_v');
for i = 1:V  % number of graphs from data (may need to put this part into graph section later)
    tmp = tmp + dim_V(i);
    dim_sum(i) = tmp;
end
dim_V_ind1 = [1,dim_sum(1:V-1)+1];
dim_V_ind2 = [dim_sum(1:V-1),dim_sum(V)];
    
tmp_rec = 0;
in_count = 1;
while in_count <= 20
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
        B = Y*C(:,dim_V_ind1(v):dim_V_ind2(v));
        nv(v) = trace(F_v{v}'*A_norm{v}*F_v{v});
    end
    
     lambda = norm(nv);
     alpha = nv./lambda;

    % ->fix alpha     
     for v = 1:V
        [F_v{v}, obj] = GPI(A_norm{v}, B, eta/alpha(v));
        %problem here 1. how to set the third para for GPI, how to make the
        %B,use all feature to do kmeans cluster or sperate each feature.
     end
    
     % calculate obj_value
    tmp = 0;
    for v = 1:V
        tmp = tmp + alpha(v)*trace(F_v{v}'*A_norm{v}*F_v{v});
    end
    obj_value(in_count) = tmp - eta * norm(cell2mat(F_v) - Y*C, 'fro')^2;
    
    if in_count == 1
        tmp_rec = obj_value(in_count);
        in_count = in_count + 1;
    elseif in_count>1 && obj_value(in_count) - tmp_rec > 0.00001*tmp_rec     
        tmp_rec = obj_value(in_count);
        in_count = in_count + 1;
    elseif obj_value(in_count) - tmp_rec < 0.00001*tmp_rec && obj_value(in_count) - tmp_rec > 0 
        break;
    end
    
    
%     if in_count == 1
%         tmp_rec = obj_value(in_count);
%         in_count = in_count + 1;
%     elseif in_count>1 && obj_value(in_count) - tmp_rec > 0.001*tmp_rec     
%         tmp_rec = obj_value(in_count);
%         in_count = in_count + 1;
%     elseif obj_value(in_count) - tmp_rec < 0.001*tmp_rec && obj_value(in_count) - tmp_rec > 0 
%         break;
%     end
end

    %         B = Y*C(dim_V_ind1(v):dim_V_ind2(v),:)';
    %         [F_v, ~] = GPI(A_norm, B, eta/alpha(v));
    %
    %         nv(v) = trace(F_v'*A_norm*F_v);
    %
    %         data_new{v} = F_v;

    
    
    
%     % fix F and alpha calculate Y and C
%     [idx, C, ~, kmeans_dist] = kmeans(X, nbclusters);
%     
%     kmeans_likelihood = 1./kmeans_dist;
%     temp1 = repmat(sum(kmeans_likelihood'),[nbclusters,1]);
%     F0 = kmeans_likelihood./temp1';  %compute the how likely xi belongs to the j-th cluster
%     
%     C = C';
%     Y = zeros(n, nbclusters);
%     
%     for i = 1:n
%         Y(i,idx(i)) = 1;
%     end
%     
%     
%     
%     % fix Y and C calculate F and alpha
%     
%     % fix F calculate alpha
%     lambda = norm(nv);
%     alpha = alpha./lambda;
%     data = data_new;
%     F = cell2mat(data);
%     
%     % calculate obj_value
%     tmp = 0;
%     for v = 1:V
%         tmp = tmp + alpha(v)*trace(data{v}'*A_norm*data{v});
%     end
%     obj_value(i) = tmp + eta * norm(F - Y*C')^2;
    


% %results display
% figure;
% [Y_ind,~,~] = find(Y');
% for i = 1: nbclusters   
%     points = data(Y_ind == i, :);
%     plot(points(:,1), points(:,2), plotchoices{i});
%     hold on;
% end





% [evectors, evalues] = eig(laplacian);
% 
% newspace = evectors(:, eigv(1,1): eigv(1,2));
% n = size(eigv);
% for i = 2: n(1)
%     
%     newspace = [newspace evectors(:, eigv(i,1): eigv(i,2))];
% end
% 
% if(strcmp(clusteralgo, 'kmean'))
%     
%     clusters = kmeans(newspace, nbclusters);
% else
%     clusters = 1 + (newspace > 0);
% end
% 
% if(dim == 2)
%     figure;
%     for i = 1: nbclusters
% 
%         points = data(clusters == i, :);
%         plot(points(:,1), points(:,2), plotchoices{i});
%         hold on;
%     end
%     title('clustered data using spectral clustering');
%     grid on;
% end