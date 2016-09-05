function sigma = determineSigma(data, type, param)

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