function [clusters, F, oobj, mobj] = algONGC(L, X, m, para, iniMethod)
%% algONGC_LinPro function
% used to compute orthogonal non-negative graph clustering with linear
% projection term (which may improve the performance)
% min_{F'*F=I, G>=0, W, b}  trace(F'*L*G) + mu*||F-G||^2_f + 
% gamma*(||WtX + b1t-F||^2_f + etag||W||^2_f)

% --- details --- (option)

% --- version ---- (option)

% --- Input ---
% X: data matrix R(d*n)
% H: centering matrix 
% L: input graph R{n*n}
% F: new representation R(c*n)
% G: the intermediate coefficient in new form, replace F in some part R(c*n) 
% W: projection matrix R(d*c)
% b: bias R(c)
% noneVec: 1(n)
% gamma, mu, etag: all paras > 0

% --- output ----
% F: the new representation F
% W: projection matrix R(d*c)
% G
% b

% --- ref ---

% --- note ---(option)

% by Lance Liu 

% parameter setting
lapmatrixchoice = 'sym';
eigv = [1 m];

%% iters setting !!!!
%niters = 30;  % for usual case
niters = 100;  % for converge analysis

%% initialisation
mu = para.mu;
gamma = para.gamma;
etag = para.etag;

I = eye(size(L));
n = size(L,1);
d = size(X,1);
oneNVec = ones(n,1);


H = eye(n) - 1/n*(oneNVec*oneNVec');
Lg = H - X'*pinv(X*X'+etag*eye(d));



if strcmp(iniMethod, 'orth_random')
% random initialisation
G = orth(rand(n,m));
F = orth(rand(n,m));

elseif strcmp(iniMethod, 'random')
%% random initialisation
G = rand(n,m);
F = rand(n,m);

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
  
%% computation
% original obj 
oobj1 = trace(F'*L*F) + gamma*(norm(W*X+b*oneNVec-F,'fro')^2+etag*norm(W,'fro')^2);

% modified obj
mobj1 = trace(F'*L*G) + mu*norm(F-G,'fro')^2 + gamma*(norm(W*X+b*oneNVec-F,'fro')^2+etag*norm(W,'fro')^2);

for i = 1:niters
    %% solve F fix G
    L1 = L*G-2*mu*G+gamma*G*Lg;
    eig_vec = eig(L1);
    alpha = max(eig_vec);
    
    M = (alpha+eps)*I - L1;
    %     M = ((alpha+10000*eps)*I - L1)*G; %for test
    [U, S, V]=svd(M, 'econ');
    F = U*V';
    F_iter{i} = F;
    
    %% solve G fix F
    P = 1/2*(1/mu*L'*F-2*F+gamma/mu*F*Lg);
    G = -P;
    G(G<0) = 0;
    G_iter{i} = G;
    
    %% solve W fix F, G, and b
    W = pinv(X*X'+etag*eye(d))*X*F';
    
    %% solve b fix F, G, and W
    b = 1/n*F*oneNVec;
    
    %% obj value
    oobj(i) = trace(F'*L*F);
    mobj(i) = trace(F'*L*G)+mu*norm(F-G,'fro');
end

oobj = [oobj1,oobj];
mobj = [mobj1,mobj];

clusters = kmeans(F, m);
