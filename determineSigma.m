function sigma = determineSigma(data, type, param)
%% determine the parameter sigma in neighbourhood graph construction
% --- details ---

% --- Input ---
% data: input data (ndim * nsample) or data is a cell 
% type: choose a method to compute sigma
% param: an empirical hyper parameter may relate to the percentage of data to compute the sigma

% --- output ----
% sigma: the parameter sigma in neighbourhood graph construction

% --- ref ---
% Survey on spectral clustering algorithms
% Ng, Andrew Y., Michael I. Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." NIPS. Vol. 14. No. 2. 2001.

% by Lance Liu 

%%
if iscell(data)
    X = cell2mat(data');
else
    X = data;
end

array = pdist(X');
dmax = max(array);
dmin = min(array);
switch type
    case 1
       sigma = param*(dmax - dmin);
    case 2
       sigma = param*dmax;
end