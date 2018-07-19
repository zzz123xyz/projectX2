function [clusters, F, oobj, mobj] = algONGC(L, m, mu)
%% algONGC function
% used to compute orthogonal non-negative graph clustering 
% min_{F'*F=I, G>=0}  trace(F'*L*G) + mu*trace(F-G)

% --- details --- (option)

% --- version ---- (option)

% --- Input ---
% L: input graph R{m*m}
% m: number of reduced dim (clusters)
% mu: parameter

% --- output ----
% F: the new representation F

% --- ref ---

% --- note ---(option)

% by Lance Liu 

%% initialisation
niters = 100;
I = eye(size(L));
n = size(L,1);
L1 = L-2*mu*I;
alpha = eigs(L1,1);

% random initialisation
G = orth(rand(n,m));
F = orth(rand(n,m));

% maybe other methods

% original obj 
oobj1 = trace(F'*L*F);

% modified obj
mobj1 = trace(F'*L*G)+mu*norm(F-G,'fro')^2;

for i = 1:niters
    %% solve F fix G
    M = (alpha*I - L1)*G;
    [U, S, V]=svd(M, 'econ');
    F = U*V';
    F_iter{i} = F;
    
    %% solve G fix F
    P = 1/2*(1/mu*L'-2*I)*F;
    G = -P;
    G(G<0) = 0;
    G_iter{i} = G;
    
    % obj value
    oobj(i) = trace(F'*L*F);
    mobj(i) = trace(F'*L*G)+mu*norm(F-G,'fro')^2;
end

clusters = kmeans(F, m);
