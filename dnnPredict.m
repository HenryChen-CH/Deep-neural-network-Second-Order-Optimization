function [ pred ] = dnnPredict( theta, layer_size, X )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
[W,b] = dnnParamToStack(theta, layer_size);
m = length(W);
output_state = cell(m,1);
soft_max = 0;
%% forward propagation
unit = W{1} * X;
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
[~,pred] = max(output_state{m});

end

