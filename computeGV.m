function gv = computeGV(theta, v, training_data, training_target, layer_size, lambda)
%UNTITLED7 Summary of this function goes here
%   This function will implement the Gauss-newton vector product. 
% Every case wis arranged in a column in training_data

%% initialize the variable
% variable used in the forward propagation
[W, b] = dnnParamToStack(theta, layer_size);
[V_w, V_b] = dnnParamToStack(v, layer_size);
case_num = length(training_target);
m = length(V_w);
R_x = cell(m,1);
R_y = cell(m,1);
y = cell(m,1);
dydx = cell(m,1);

G_w = cell(m,1);
G_b = cell(m,1);

%% R operator forward propagation
R_x{1} = V_w{1} * training_data;
R_x{1} = bsxfun(@plus,R_x{1},V_b{1}(:));
unit = W{1} * training_data;
unit = bsxfun(@plus, unit, b{1}(:));
y{1} = sigmoid(unit);
dydx{1} = y{1}.*(1-y{1});

R_y{1} = R_x{1} .* dydx{1};
for i = 2:m
    R_x{i} = W{i} * R_y{i-1} + V_w{i} * y{i-1};
    R_x{i} = bsxfun(@plus,R_x{i},V_b{i}(:));
    
    %normal propagation
    unit = W{i} * y{i-1};
    unit = bsxfun(@plus,unit,b{i}(:));
    y{i} = sigmoid(unit);
    dydx{i} = y{i}.*(1-y{i});
    
    R_y{i} = R_x{i}.*dydx{i};
end;

%% back propagation
delta = R_y{m};
%normal back propagation
for i = m:-1:2
    G_w{i} = (delta * y{i-1}')./case_num;
    G_b{i} = sum(delta,2)./case_num;
    
    delta = W{i}'*delta .* dydx{i-1};
end;

G_w{1} = (delta * training_data')./case_num;
G_b{1} = sum(delta,2)./case_num;


%% convert to vector
gv = dnnParamToVec(G_w,G_b,layer_size);

%damping
gv = gv + (lambda./length(training_target)) * v;

end



