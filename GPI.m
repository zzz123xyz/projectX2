% max_{X'*X=I}  trace(X'*A*X) + 2*r*trace(X'*B)
% A must be positive semi-definite
function [X, obj] = GPI(A, B, r)

NITER = 100;
[n,m] = size(B);
X = orth(rand(n,m));

for iter = 1:NITER
    M = A*X + r*B;
    [U,~,V] = svd(M,'econ');
    X = U*V';
    
    obj(iter,1) = trace(X'*A*X) + 2*r*trace(X'*B);
end;