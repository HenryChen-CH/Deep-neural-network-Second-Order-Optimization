clear
load training_data

%% initialize parameters for the neural network
input_layer_size = size(training_data,2);
output_layer_size = 10;
layer_size = [input_layer_size,500,300,30,output_layer_size];
theta = dnnRandInitializeWeights(layer_size);
[W,b] = dnnParamToStack(theta,layer_size);
lambda = 4;

%% training using hessian-free
options.epoches = 50;
options.max_iterations = 250;

computeHv = @computeGV;
[theta,costs] = minFunHessianFree(theta,computeHv,training_data',training_target,layer_size);
plot(1:length(costs),costs);

%% Training data
training_cost = dnnCostOnly(theta,training_data',training_target,layer_size,0);
training_pred = dnnPredict(theta,layer_size,training_data');
fprintf('Training data Cost:%.4f Accuracy: %.4f \n',training_cost,mean(training_pred(:)==training_target(:))*100);

%% CV
CV_cost = dnnCostOnly(theta,CV_data',CV_target,layer_size,0);
CV_pred = dnnPredict(theta,layer_size,CV_data');
fprintf('CV data Cost:%.4f Accuracy:%.4f \n',CV_cost,mean(CV_pred(:)==CV_target)*100);