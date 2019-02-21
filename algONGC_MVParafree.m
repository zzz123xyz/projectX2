function [clusters, F, oobj, mobj] = algONGC_MVParafree(data, nbclusters, mu, method, param, iniMethod)
%% algONGC_MVParafree function
% used to compute orthogonal non-negative graph clustering with parameter
% free method
% min_{F'*F=I, G>=0} \sum sqrt(trace(F'*L*G)) + mu*trace(F-G)

% --- details --- (option)

% --- version ---- (option)

% --- Input ---
% data: data: X feature matrix dim: R^{m*n} (m features & n samples)
% nbclusters: number of reduced dim (clusters)
% mu: the only parameter
% method: graph construction method
% param: parameters of the graph construction method
% iniMethod: initialization method for F and G

% --- output ----
% F: the new representation F

% --- ref ---

% --- note ---(option)

% by Lance Liu 

%% parameter setting !!!
lapmatrixchoice = 'sym'; %laplace matrix construction method. options: unormalized, sym(default), rw  
eigv = [1 nbclusters];

%% iters setting !!!!
niters = 30;  % for usual case
% niters = 100;  % for converge analysis

V = numel(data); % the data is V views
n = size(data{1},2); % the there n data ()

%% initialize alpha
alpha = ones(V)*1/V;

%% construct graph or view
for v = 1:V
    X = data{v};
    
    if isscalar(param)
        paramOne = param;
    else
        paramOne = param(v);
    end
    [A_norm{v}, paramOne] = constructGraph(X, nbclusters, method, paramOne);
    
%     [U_,S,~] = svd(A_norm{v}, 'econ');
%     U = U_(:,  eigv(1,1)+1: eigv(1,2)+1);
%     
%     sq_sum = sqrt(sum(U.*U, 2)) + 1e-20;
%     F_v{v} = U ./ repmat(sq_sum, 1, nbclusters);
    
    %F_v{v} = U;
end

%% initialize the F and G
if strcmp(iniMethod, 'orth_random')
% orthognal initialisation
G = orth(rand(n,nbclusters));
F = orth(rand(n,nbclusters));

elseif strcmp(iniMethod, 'random')
%% random initialisation
G = rand(n,nbclusters);
F = rand(n,nbclusters);

% elseif strcmp(iniMethod, 'SPCL')
% %% SPCL initialisation
%   dmat = diag(sum(L, 2));
%     switch lapmatrixchoice
%         case 'unormalized'
%             %A_norm{v} = dmat - wmat;
%             A2_norm = L;
%         case 'sym'
%             %A_norm{v} = eye(nbsamples) - (dmat^-0.5) * wmat * (dmat^-0.5);
%             A2_norm = (dmat^-0.5) * L * (dmat^-0.5);
%         case 'rw'
%             %A_norm{v} = eye(nbsamples) - (dmat^-1) * wmat;
%             A2_norm = (dmat^-1) * L;
%     end
%     [U_,S,~] = svd(A2_norm, 'econ');
%     % Unormalize on each row
%     U = U_(:,  eigv(1,1)+1: eigv(1,2)+1);
%     %F_v{v} = U;
%     % Normalize each row to be of unit length
%     sq_sum = sqrt(sum(U.*U, 2)) + 1e-20;
%     F = U ./ repmat(sq_sum, 1, nbclusters);
%     G = F;
end
  
%% compute the obj first time
% original obj 
temp = cellfun(@(x) sqrt(trace(F'*x*F)), A_norm, 'UniformOutput', false);
oobj1 = sum([temp{:}]);

% modified obj
temp = cellfun(@(x) sqrt(trace(F'*x*G)), A_norm, 'UniformOutput', false);
mobj1 = sum([temp{:}]) + mu*norm(F-G,'fro')^2;

% modified with alpha obj
temp = cellfun(@(x) trace(F'*x*G), A_norm, 'UniformOutput', false);
maobj1 = sum(alpha.*[temp{:}]) + mu*norm(F-G,'fro')^2;

%% start the iterations
for i = 1:niters
    %% update the joint graph A according to the alpha
    A = [];
    for v = 1:V
        A = A + alpha(v)*A_norm{v}; %*****
    end

    %% initialisation of the iterations
    I = eye(size(A));
    n = size(A,1);
    L1 = A-2*mu*I; 
    eig_vec = eig(L1);
    beta = max(eig_vec); %I want to obtain the value of beta before 
                          %the whole iteration procedure to save computation time cost
    
    %% solve F fix G and alpha
    M = ((beta+eps)*I - L1)*G; % see the alpha part in initialization 
%     M = ((alpha+10000*eps)*I - L1)*G; %for test
    [U, S, V]=svd(M, 'econ');
    F = U*V';
    F_iter{i} = F;
    
    %% solve G fix F and alpha
    tmp = [];
    for v = 1:V
        tmp = tmp + alpha(v) * A_norm{v}; %*****
    end
    P = 1/2*(1/mu * tmp - 2*I)*F;
    G = -P;
    G(G<0) = 0;
    G_iter{i} = G;
    
    %% solve alpha fix F and G
    alpha = cellfun(@(x) 1/(2*sqrt(trace(F'*x*F))), A_norm, 'UniformOutput', false);
    alpha = [alpha{:}];
    
    % obj value
    temp = cellfun(@(x) sqrt(trace(F'*x*F)), A_norm, 'UniformOutput', false);
    oobj(i) = sum([temp{:}]);
    
    temp = cellfun(@(x) sqrt(trace(F'*x*G)), A_norm, 'UniformOutput', false);
    mobj(i) = sum([temp{:}]) + mu*norm(F-G,'fro')^2;
    
    temp = cellfun(@(x) trace(F'*x*G), A_norm, 'UniformOutput', false);
    maobj(i) = sum(alpha.*[temp{:}]) + mu*norm(F-G,'fro')^2;
    
end

oobj = [oobj1,oobj];
mobj = [mobj1,mobj];
maobj = [maobj1,maobj];

clusters = kmeans(F, nbclusters);
