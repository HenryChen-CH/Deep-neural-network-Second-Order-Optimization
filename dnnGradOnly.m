function [ grad ] = dnnGradOnly( theta, training_data, training_target, layer_size,lambda )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
% use forward and backword propagation to compute the cost function and
% gradient. 
% Every case is arranged in a column in the training_data
%% initialize the parameter
[W,b] = dnnParamToStack(theta, layer_size);
% m equals number of layers - 1
m = length(W);
output_state = cell(m,1);
W_grad = cell(m,1);
b_grad = cell(m,1);
case_num = length(training_target);
%determine the last layer is a soft-max or logistic
soft_max = 0;

%% forward propagation
unit = W{1} * training_data;
unit = bsxfun(@plus, unit, b{1}(:));
output_state{1} = sigmoid(unit);
for i = 2 : m - 1
    unit = W{i} * output_state{i - 1};
    unit = bsxfun(@plus, unit, b{i}(:));
    output_state{i} = sigmoid(unit);
end;
if soft_max == 1
    % the last layer is a softmax layer
    unit = W{m} * output_state{m - 1};
    unit = bsxfun(@plus, unit, b{m}(:));
    unit = exp(unit);
    unit_sum = sum(unit,1);
    output_state{m} = bsxfun(@rdivide, unit, unit_sum);
else
    %the last layer is a logistic unit
    unit = W{m} * output_state{m - 1};
    unit = bsxfun(@plus, unit, b{m}(:));
    output_state{m} = sigmoid(unit);
end;

% The error of the last layer
expended_matrix = eye(layer_size(end));
expended_target = expended_matrix(:,training_target);
output_delta = output_state{m} - expended_target;


%% backpropagation
delta = output_delta;
W_grad{m} = (delta * output_state{m-1}')./case_num;
b_grad{m} = sum(delta,2)./case_num;
for i = (m-1):-1:2
    delta = (W{i+1}'* delta).*output_state{i}.*(1-output_state{i});
    W_grad{i} = (delta * output_state{i-1}')./case_num;
    b_grad{i} = sum(delta,2)./case_num;
end;
delta = (W{2}'*delta).*output_state{1}.*(1-output_state{1});
W_grad{1} = (delta * training_data')./case_num;
b_grad{1} = sum(delta,2)./case_num;

%regularization
if lambda ~= 0
    for i = 1:m
        W_grad{i} = W_grad{i} + lambda./case_num * W{i};
    end;
end;

%% convert to vector
grad = dnnParamToVec(W_grad,b_grad,layer_size);

