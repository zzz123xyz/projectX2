function A = ULGE(X, method, m, r, k, p)
%% ULGE function (Unsupervised Large Graph Embedding)
% used for Similarity Matrix Construction with Anchor-based Strategy

% --- details --- (option)

% --- version ---- (option)

% --- Input ---
% X: X data matrix
% method: what method to use (kmeans or random)
% m: number of anchor points
% r: downsampling ratio (differet setting for different datasets)
% k: number of nn for Anchor-based Similarity Matrix construction (default 5)
% p: the number of reduced rank

% --- output ----
% F: the new representation F

% --- ref ---

% --- note ---(option)
% ATTENTION !!!
% the A here is the dimension reduced graph matrix A_star (A* in paper ULGE)
% not the original A1 (A in paper ULGE)

% by Lance Liu 

%%
if nargin <5
   p = 10;   % 5 nn to construct graph in paper ULGE
end

if nargin <4
   k = 5;   % 5 nn to construct graph in paper ULGE
end

issymmetric = 1;

switch method
    case 'kmeans'
        n = size(X,1); % number of samples in dataset
        idxVec = randsample(n, n/r);
        X_sampled = X(idxVec,:);
        [Uind, U] = kmeans(X_sampled, m);
        
    case 'random'
        
end

% !!! problem here, add anchor parts 

Z = constructW_PKNA(X', U', k, issymmetric);

D = diag(sum(Z,2));
A1 = Z*D^(-1)*Z';

B = Z*D^(-1/2);
[Fp, Sp, ~] = svds(B, p);
A_star = Fp*(Sp*Sp')*Fp';

A = A_star; % the A here is the dimension reduced graph matrix A_star 
            % (A* in paper ULGE) not the original A1 (A in paper ULGE)
            
A = (A+A')/2; % make the constructed graph symmetric to avoid small 
                %computational turbulence which cause symmetric elements
