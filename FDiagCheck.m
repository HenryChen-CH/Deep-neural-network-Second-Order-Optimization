clear
load training_data;
input_layer_size = size(training_data,2);
output_layer_size = 10;
layer_size = [input_layer_size,500,250,100,output_layer_size];
theta = dnnRandInitializeWeights(layer_size);
[W,b] = dnnParamToStack(theta,layer_size);
lambda = 0;
 

training_data = training_data(1:1000,:);
training_target =  training_target(1:1000);

%% F diag check
t0 = clock;
grad =dnnGradOnly( theta, training_data', training_target, layer_size, lambda );
F_diag_1 = computeFDiag(theta,training_data',training_target,layer_size);
t1 = clock;
F_diag_2 = zeros(size(F_diag_1));
for i = 1:length(training_target)
    grad_i =dnnGradOnly( theta, training_data(i,:)', training_target(i), layer_size, lambda );
    F_diag_2 = F_diag_2 + grad_i.^2;
end;
t2 = clock;
disp([F_diag_1 F_diag_2 F_diag_1-F_diag_2]);
disp(var(F_diag_1-F_diag_2));
disp(etime(t1,t0));
disp(etime(t2,t1));