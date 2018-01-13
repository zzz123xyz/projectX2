function [clusters, F, oobj, mobj] = algONGC(L, m, mu, iniMethod)
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
lapmatrixchoice = 'sym';
eigv = [1 m];
niters = 100;
I = eye(size(L));
n = size(L,1);
L1 = L-2*mu*I;
alpha = eigs(L1,1);

if strcmp(iniMethod, 'random')
%% random initialisation
G = orth(rand(n,m));
F = orth(rand(n,m));

elseif strcmp(iniMethod, 'SPCL')
%% SPCL initialisation
  dmat = diag(sum(L, 2));

    switch lapmatrixchoice
        case 'unormalized'
            %A_norm{v} = dmat - wmat;
            A_norm = L;
        case 'sym'
            %A_norm{v} = eye(nbsamples) - (dmat^-0.5) * wmat * (dmat^-0.5);
            A_norm = (dmat^-0.5) * L * (dmat^-0.5);
        case 'rw'
            %A_norm{v} = eye(nbsamples) - (dmat^-1) * wmat;
            A_norm = (dmat^-1) * L;
    end

    [U_,S,~] = svd(A_norm, 'econ');
    
    % Unormalize on each row
    U = U_(:,  eigv(1,1)+1: eigv(1,2)+1);
    %F_v{v} = U;
    
    % Normalize each row to be of unit length
    
    sq_sum = sqrt(sum(U.*U, 2)) + 1e-20;
    F = U ./ repmat(sq_sum, 1, m);
    
    G = F;
end
  
% original obj 
oobj1 = trace(F'*L*F);

% modified obj
mobj1 = trace(F'*L*G)+mu*norm(F-G,'fro');

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
    mobj(i) = trace(F'*L*G)+mu*norm(F-G,'fro');
end

clusters = kmeans(F, m);
