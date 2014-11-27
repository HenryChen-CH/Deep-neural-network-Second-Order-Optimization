function [ theta, costs ] = dnnHessianFreeOpt( initial_theta, training_data, training_target, layer_size,options )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% this implement the hessian free optimizator

%% Initialize the parameters
max_epoches = options.max_iterations;
theta = initial_theta;
[W,b] = dnnParamToStack(theta, layer_size);

%% Start hessian-free optimization
for epoch = 1:max_epoches
end;


end







