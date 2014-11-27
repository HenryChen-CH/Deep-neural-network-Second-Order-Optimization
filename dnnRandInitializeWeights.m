function [ theta ] = dnnRandInitializeWeights( layer_size )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% This function help initialize the weight and bias of the neural network

% total number of parameters including W and b
layer_num = length(layer_size);
m = layer_num - 1;
W = cell(m,1);
b = cell(m,1);

%% initialize the parameters
for i = 1:m
    r = sqrt(6./(layer_size(i)+layer_size(i+1)));
    W{i} = rand(layer_size(i+1),layer_size(i)) * 2 * r - r;
    b{i} = zeros(layer_size(i+1),1);    
end;
theta = dnnParamToVec(W,b,layer_size);

% layer_size = layer_size(:);
% param_num = layer_size(1:(end-1))'*layer_size(2:end)+sum(layer_size(2:end));
% 
% ep = 0.2;
% theta = rand(param_num,1)*2*ep -ep;

end

