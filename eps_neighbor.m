function W = eps_neighbor(X, )

X = X'; % transpose to observations in rows for pdist computation
D = pdist2(X);
W = D>