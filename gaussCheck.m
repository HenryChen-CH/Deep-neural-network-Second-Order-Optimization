clear
load training_data;
input_layer_size = size(training_data,2);
output_layer_size = 10;
layer_size = [input_layer_size,500,200,output_layer_size];
theta = dnnRandInitializeWeights(layer_size);
[W,b] = dnnParamToStack(theta,layer_size);
lambda = 0;
 

training_data = training_data(1:5000,:);
training_target =  training_target(1:5000);

%% compute using R operator
v =dnnGradOnly( theta, training_data', training_target,layer_size,lambda );
gv_1 = computeGV(theta, v, training_data', training_target, layer_size,0);
gv_2 = computeGV_1(theta, v, training_data', training_target, layer_size,0);
gv_3 = computeGV_2(theta, v, training_data', training_target, layer_size,0);
hv_1 = computeHV(theta,v,training_data',training_target,layer_size,0);

disp(var(gv_1-gv_2));
disp(var(gv_2-gv_3));
disp(var(gv_1 - hv_1));