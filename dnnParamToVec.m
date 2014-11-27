function [ theta ] = dnnParamToVec( W,b,layer_size )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Convert the W and b to a vector
layer_size = layer_size(:);
param_num = layer_size(1:(end-1))'*layer_size(2:end)+sum(layer_size(2:end));
theta = zeros(param_num,1);
m = length(W);
p = 1;
for i = 1:m
    len = length(W{i}(:));
    p_next = p + len;
    theta(p:(p_next-1)) = W{i}(:);
    p = p_next;
    p_next = p + length(b{i});
    theta(p:(p_next-1)) = b{i};
    p = p_next;
end;

end

