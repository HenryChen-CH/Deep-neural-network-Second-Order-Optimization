clear
load training_data;
input_layer_size = size(training_data,2);
output_layer_size = 10;
layer_size = [input_layer_size,200,500,output_layer_size];
theta = dnnRandInitializeWeights(layer_size);
[W,b] = dnnParamToStack(theta,layer_size);
lambda = 0;
 

training_data = training_data(1:5000,:);
training_target =  training_target(1:5000);


%% compute hessian vector numerically
v =dnnGradOnly( theta, training_data', training_target, lambda, layer_size );
epsilion = 1e-5;
grad_1 = dnnGradOnly( theta-epsilion*v, training_data', training_target, lambda, layer_size );
grad_2 = dnnGradOnly( theta+epsilion*v, training_data', training_target, lambda, layer_size );
hv_1 = (grad_2-grad_1)./2./epsilion;

%% compute using R operator

hv_2 = computeHV(theta, v, training_data', training_target, layer_size);
dif = norm(hv_2-hv_1);
disp([hv_1 hv_2 hv_2-hv_1]);
disp(dif);