load training_data

%% initialize parameters for the neural network
input_layer_size = size(training_data,2);
output_layer_size = 10;
layer_size = [input_layer_size,2,10,output_layer_size];
theta = dnnRandInitializeWeights(layer_size);
[W,b] = dnnParamToStack(theta,layer_size);
lambda = 4;
%% gradient check
training_data = training_data';
training_data = training_data(:,1:2000);
training_target = training_target(1:2000);
funObj = @(theta)dnnCostFunction(theta,training_data,training_target,0,layer_size);
e=gradientCheck(theta, funObj);
disp(e);