function [ F_diag ] = computeFDiag( theta, training_data,training_target,layer_size)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%% Initialize the parameter
[W,b] = dnnParamToStack(theta,layer_size);
m = length(W);
y = cell(m,1);
dydx = cell(m,1);
F_w = cell(m,1);
F_b = cell(m,1);

%% forward propagation
unit = W{1} * training_data;
unit = bsxfun(@plus, unit, b{1}(:));
y{1} = sigmoid(unit);
dydx{1} = y{1}.*(1-y{1});

for i = 2 : m
    unit = W{i} * y{i - 1};
    unit = bsxfun(@plus, unit, b{i}(:));
    y{i} = sigmoid(unit);
    dydx{i} = y{i}.*(1-y{i});
end;


% The error of the last layer
expended_matrix = eye(layer_size(end));
expended_target = expended_matrix(:,training_target);
delta = y{m} - expended_target;

%% back propagation
for i = m:-1:2
    delta_2 = delta.^2;
    F_w{i} = delta_2 * (y{i-1}.^2)';
    F_b{i} = sum(delta_2,2);
    delta = (W{i}'* delta).*dydx{i-1};
end;

delta_2 = delta.^2;
F_w{1} = delta_2*(training_data.^2)';
F_b{1} = sum(delta_2,2);

%% convert to vector
F_diag = dnnParamToVec(F_w,F_b,layer_size);

end

