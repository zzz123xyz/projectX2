function W = constructW_PKNA(X, U, k, m)

%% constructW_PKNA function (modified from constructW_PKN)
% construct similarity matrix with probabilistic k-nearest neighbors. 
% It is a parameter free, distance consistent similarity.

% --- details --- (option)

% --- version ---- (option)

% --- Input ---
% X: each column is a data point
% U: anchor matrix
% k: number of neighbors
% m: number of anchors
% issymmetric: set W = (W+W')/2 if issymmetric=1
% W: similarity matrix

% --- output ----
% F: the new representation F

% --- ref ---

% --- note ---(option)

% by Lance Liu 

if nargin < 2
    k = 5;
end

[dim, n] = size(X);
D = L2_distance_1(X, U);
[dumb, idx] = sort(D, 2); % sort each row

W = zeros(n,m);
for i = 1:n
    id = idx(i,:);% it's different from constructW_PKN, here is to compute
    %anchor point. In constructW_PKN is to compute normal nearest neighbors.
    di = D(i, id);
    W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps); %in (di(k+1)-di), 
    %the last k+1 elemet is substracted to 0, so has no effect on final
    %graph
end


% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
function d = L2_distance_1(a,b)
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b



if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

d = real(d);
d = max(d,0);

% % force 0 on the diagonal? 
% if (df==1)
%   d = d.*(1-eye(size(d)));
% end





