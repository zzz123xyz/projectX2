function predict = algONGC_MVParafree_GC_linpro_OOS(data, F, W, b, A) % **** try this first for OOS 

if iscell(data)
    allData = cell2mat(data')';
else
    allData = data'; % allData = data'; for cnn feature?
end
allData = allData';
nsample = size(allData,2);
one_n = ones(nsample, 1);
[~, R] = MSC_w_F(F,20,A);
Y = W'*allData + b*one_n'; % according to the SEC paper p7 left column
Y_tilt = R'*normc(Y);

[p_value, predict] = max(Y_tilt); % now the problem is how to evaluate the results
predict = predict';
% can still use the clustering evalution method, acc F1 score !! should I
% compared with SEC???

% reuslt = ClusteringMeasure(label, predict);