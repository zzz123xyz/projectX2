function [clusters, F, oobj, mobj] = algONGC_MVParafree_GC(data, nbclusters, mu, method, param, iniMethod)
%% algONGC_MVParafree_GC function
% ---description---
% algorithm orthognal non-negative graph clustering with multiview
% parameter free (GC means graph clustering indicate the new version of 
% graph clustering using parameter auto optimization technique not the sqrt 
% optimization technique as the preivious idea in ONGC_LinPro_sqrt_multiview.docx)

% min_{F'*F=I, G>=0, \alpha'*vec(1)=1, \alpha>=0 } \sum 
% trace(sum(\alpha_v*A_v) - FG') + mu*trace(F-G)

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
% niters = 30;  % for usual case
niters = 100;  % for converge analysis

nV = numel(data); % the data is V views
n = size(data{1},2); % the there n data ()

%% initialize alpha
alpha = ones(1, nV)*1/nV;

%% construct graph or view
A_norm = cell(1,nV);
for v = 1:nV
    X = data{v};
    
    if isscalar(param)
        paramOne = param;
    elseif iscell(param) 
        paramOne = param;
    else
        paramOne = param(v);
    end
    [A_norm{v}, ~] = constructGraph(X, nbclusters, method, paramOne);
   
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
oobj1 = compute_original_obj(alpha, A_norm, F);

% modified with obj
mobj1 = compute_modified_obj(alpha, A_norm, F, G, mu);

%% start the iterations
G_iter = cell(1,niters);
F_iter = cell(1,niters);
oobj = zeros(1,niters);
mobj = zeros(1,niters);
for i = 1:niters
%     %% update the joint graph A according to the alpha
%     A = zeros(size(A_norm{1}));
%     for v = 1:nV
%         A = A + alpha(v)*A_norm{v}; 
%     end
% 
%     %% initialisation of the iterations
%     I = eye(size(A));
%     n = size(A,1);
                          
    %% solve G fix F and alpha
    A_combine = compute_combined_graph(alpha, A_norm);
    G = (A_combine*F+mu*F)/(1+mu);
    G(G<0) = 0;
    G_iter{i} = G;
     
    %% solve F fix G and alpha 
    M = A_combine*G+mu*G; % see the alpha part in initialization 
%     M = ((alpha+10000*eps)*I - L1)*G; %for test
    [U, S, V]=svd(M, 'econ');
    F = U*V';
    F_iter{i} = F;
        
    %% solve alpha fix F and G
    B = cell2mat(cellfun(@(x) x(:), A_norm, 'UniformOutput', false));
    tmp = F*G';
    C = tmp(:);
    W = inv(B'*B);
    Vv = B'*C;
    One_v = ones(nV,1);
    N = One_v'*W*One_v;
    J = W*Vv+W*One_v/N-W*(One_v*One_v')*W/N;
    J(J<0) = 0;
    alpha = J;
  
    % obj value
    oobj(i) = compute_original_obj(alpha, A_norm, F);
    mobj(i) = compute_modified_obj(alpha, A_norm, F, G, mu);

end

oobj = [oobj1,oobj];
mobj = [mobj1,mobj];

clusters = kmeans(F, nbclusters);
end

function obj = compute_original_obj(alpha, A_norm, F)
    A_combine = compute_combined_graph(alpha, A_norm);
    nsample = size(A_combine, 1);
    L = eye(nsample) - A_combine;
    % simlilar to the setting in code ONGC_linPro graph construction the
    % LA_norm is the laplacian matrix L in derivative
    obj = trace(F'*L*F);
end

function obj = compute_modified_obj(alpha, A_norm, F, G, mu)
    A_combine = compute_combined_graph(alpha, A_norm);
    obj = norm(A_combine-F*G', 'fro')^2 + mu * norm(F - G, 'fro')^2;
end

function A_combine = compute_combined_graph(alpha, A_norm)
    nV = numel(A_norm);
    A_combine = zeros(size(A_norm{1}));
    for i=1:nV
        A_combine = A_combine + alpha(i)*A_norm{i};
    end
end