function [ W, b ] = dnnParamToStack( theta, layer_size )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
% theta is the weight and bias arranged in a vector including the input
% layer, output layer and hidden layer.
% theta is a vector, including W and b

%% initialize the parameters
% first rearrange the theta vector into matrix
layer_num = length(layer_size);
W = cell(layer_num-1,1);
b = cell(layer_num-1,1);
p = 1;
for i = 1:(layer_num - 1)
    p_next = p + layer_size(i) * layer_size(i+1);
    W{i} = reshape(theta(p:p_next-1),layer_size(i+1),layer_size(i));
    p = p_next;
    p_next = p + layer_size(i+1);
    b{i} = reshape(theta(p:p_next-1),layer_size(i+1),1);
    p = p_next;
end;
    

end

