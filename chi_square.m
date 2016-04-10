function weight = chi_square(xi, xj, sigma) %****



weight = exp(-sum(((xi - xj) .^2) ./(xi+xj+eps), 2));
