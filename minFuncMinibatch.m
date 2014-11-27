function [ theta,costs ] = minFuncMinibatch( funObj,initial_theta,data,target,options )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
% This is a optimization function for neural network
% funObj is the function that compute the cost and gradient for the data and target
% [cost, grad] = funObj(...);
% initial_theta is a vector and theta is a vector of the same size as initial_theta
% option is a struct containing:iterations,momentum,learning_rate,momentum 
% options:{iterations, momentum, learning_rate, batch_size}
% batchSize and momentum is the parameter for the mini-batch gradient
% descent with momentum

%% intial the parameters for the optimization
epoches = options.iterations;
momentum = options.momentum;
learning_rate = options.learning_rate;
batch_size = options.batch_size;

velocity = zeros(size(initial_theta));

% compute the number of baches
num = length(target);
batch_num = floor(num ./ batch_size);
rand_order = randperm(num);
theta = initial_theta;
it = 0;

costs = zeros(epoches,1);

%% start the optimization
for e = 1: epoches
    for i = 1 : batch_num
        it = it + 1;
        batch_data = data(rand_order((i-1)*batch_size+1:i*batch_size),:);
        batch_target = target(rand_order((i-1)*batch_size+1:i*batch_size));
        % The data is transposed because in dnn every case should in a
        % column. I did not implement the code well, so sometimes you need
        % to transpose x
        [cost,grad] = funObj(theta,batch_data',batch_target);
        
        velocity(:) = momentum .* velocity(:) + learning_rate .* grad;
        theta(:) = theta(:) - velocity(:);
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
    end;
    
    % track the changing of the cost
    [cost,~] = funObj(theta,data',target);
    costs(e) = cost;
end;
end

