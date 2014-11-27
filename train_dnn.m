load training_data

%% initialize parameters for the neural network
input_layer_size = size(training_data,2);
output_layer_size = 10;
layer_size = [input_layer_size,500,250,30,output_layer_size];
theta = dnnRandInitializeWeights(layer_size);
[W,b] = dnnParamToStack(theta,layer_size);
lambda = 4;

%% training using mini-batch
options.iterations = 100;
options.momentum = 0.9;
options.learning_rate = 0.1;
options.batch_size =  5000;
costFunction = @(t,x,y)dnnCostFunction(t,x,y,lambda,layer_size);
[theta,costs] = minFuncMinibatch(costFunction,theta,training_data,training_target,options);
plot(1:length(costs),costs);

%% CV
CV_data = CV_data';
pred = dnnPredict(theta,layer_size,CV_data);
fprintf('The accuracy of CV data is %.4f \n',mean(pred(:)==CV_target)*100);