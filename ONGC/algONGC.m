function F = algONGC(L, mu)
%% algONGC function
% used to compute orthogonal non-negative graph clustering 

% --- details --- (option)

% --- version ---- (option)

% --- Input ---
% L: input graph R{m*m}
% mu: parameter

% --- output ----
% F: the new representation F

% --- ref ---

% --- note ---(option)

% by Lance Liu 

%% initialisation
% random initialisation
G = orth(rand(n,m));
I = eye(size(L));


%maybe other methods

%% solve F fix G 
L1 = L-2*mu*I;
alpha = eig(L1,1);
M = alpha*I - L1*G;