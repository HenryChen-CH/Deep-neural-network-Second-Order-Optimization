function [ cost ] = dnnCostOnly( theta, training_data, training_target, layer_size,lambda )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% This function only do forward propagation without back propagation to
% derive the gradient.
% Every case is arranged in a column in the training_data
%% initialize the parameter
[W,b] = dnnParamToStack(theta, layer_size);
% m equals number of layers - 1
m = length(W);
output_state = cell(m,1);
case_num = length(training_target);
%determine the last layer is a soft-max or logistic
soft_max = 0;
tolerance  = 1e-10;
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
%compute the cost function
expended_matrix = eye(layer_size(end));
expended_target = expended_matrix(:,training_target);
cost = -sum(sum(expended_target.*log(output_state{m}+tolerance) + (1-expended_target).*log(1-output_state{m}+tolerance)));
cost = cost./case_num;
% regularization
if lambda ~= 0
    for i = 1 : m
        cost = cost + lambda./2./case_num * (sum(sum(W{i}.^2)));
    end;
end;


end

