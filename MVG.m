function [C_f, Y_f, obj_value, F_vf] = MVG (data, nbclusters, eigv, method, param)

niters = 50;
eta = 0.01; %a fixed para
in_iters = 9; %increase iters

V = numel(data);
n = size(data{1},2);

for v = 1:V
    X = data{v};
    A_norm{v} = constructGraph(X, nbclusters, method, param);
    
    [U_,S,~] = svd(A_norm{v}, 'econ');
    U = U_(:,  eigv(1,1)+1: eigv(1,2)+1);
    
    F_v{v} = U;
end

tmp = 0;
[~, dim_V] = cellfun(@size, F_v');
for i = 1:V  % number of graphs from data (may need to put this part into graph section later)
    tmp = tmp + dim_V(i);
    dim_sum(i) = tmp;
end
dim_V_ind1 = [1,dim_sum(1:V-1)+1];
dim_V_ind2 = [dim_sum(1:V-1),dim_sum(V)];
    
max_rec = -inf;
count = 1;
in_count = 0;

while count <= niters
% fix F and alpha
    F_a = cell2mat(F_v);
    [idx, C] = kmeans(F_a, nbclusters); %change the dim of C into C_v (cluster according to each graph)
    
    Y = zeros(n, nbclusters);
    for j = 1:n
        Y(j,idx(j)) = 1;
    end
    
% fix Y and C   
    % ->fix F
       %B = Y*C;
    for v = 1:V
        B = Y*C(:,dim_V_ind1(v):dim_V_ind2(v));
        nv(v) = trace(F_v{v}'*A_norm{v}*F_v{v});
    end
    
     lambda = norm(nv);
     alpha = nv./lambda;

    % ->fix alpha     
     for v = 1:V
        [F_v{v}, obj] = GPI(A_norm{v}, B, eta/alpha(v));
        %problem here 1. how to set the third para for GPI, how to make the
        %B,use all feature to do kmeans cluster or sperate each feature.
     end
    
% calculate obj_value
    tmp = 0;
    for v = 1:V
        tmp = tmp + alpha(v)*trace(F_v{v}'*A_norm{v}*F_v{v});
    end
    obj_value(count) = tmp - eta * norm(cell2mat(F_v) - Y*C, 'fro')^2;
    
    if obj_value(count) > max_rec
        max_rec = obj_value(count);
        F_vf = F_v; % here _f means final out put value
        C_f = C;
        Y_f = Y;
        in_count = in_count + 1; %in_count means increasing count
    end
    
    if in_count > in_iters
       break; 
    end
    
    disp('iters:');
    disp(count);
    count = count + 1;
    
end