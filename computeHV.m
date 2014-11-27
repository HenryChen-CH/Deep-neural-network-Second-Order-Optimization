function hv = computeHV(theta, v, training_data, training_target, layer_size,lambda)
% This function will use R operator to fast compute the product of Hessian
% matrix and vector v

% variable related to R operator
%used in forwrad propagation
[V_w, V_b] = dnnParamToStack(v,layer_size);
case_num = length(training_target);
m = length(V_w);
R_x = cell(m,1);
R_y = cell(m,1);
%used when back propagation
R_w = cell(m,1);
R_b = cell(m,1);
% variable related to normal forward propagation
[W,b] = dnnParamToStack(theta, layer_size);
y = cell(m,1);%output of each layer
dydx = cell(m,1);

%% applying R operator forward propagation
%the input layer is different R=0
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


%% R operator back propagation
% the output layer is different
expension_matrix = eye(layer_size(end));
expended_target = expension_matrix(:,training_target);
R_dEdy = (expended_target./(y{m}.^2)+(1-expended_target)./((1-y{m}).^2)).*R_y{m};
dEdy = -expended_target./y{m} + (1-expended_target)./(1-y{m});

for i = m:-1:2
    R_dEdx = R_dEdy.*y{i}.*(1-y{i}) + R_x{i}.*(1-2*y{i}).*dydx{i}.*dEdy;
    dEdx = dEdy.*dydx{i};
    R_w{i} = (R_dEdx * y{i-1}' + dEdx * R_y{i-1}')./case_num;
    R_b{i} = sum(R_dEdx,2)./case_num;
    
    dEdy = W{i}'*dEdx;
    R_dEdy = W{i}'*R_dEdx + V_w{i}'*dEdx;
end;

R_dEdx = R_dEdy.*y{1}.*(1-y{1}) + R_x{1}.*(1-2*y{1}).*dydx{1}.*dEdy;
R_w{1} = R_dEdx * training_data';
R_w{1} = R_w{1}./case_num;
R_b{1} = sum(R_dEdx,2)./case_num;

%% convert to vector
hv = dnnParamToVec(R_w,R_b,layer_size);
%damping
hv = hv + lambda./length(training_target)*v;

end

