function [ theta, costs ] = minFunHessianFree(initial_theta,computeHv, training_data, training_target, layer_size)
%UNTITLED8 Summary of this function goes here
%   This function implement the hessian free optimization

%% initialize the parameters
%epoches = options.epoches;
%max_iterations = options.max_iterations;
epoches = 10;
max_iterations = 6;
theta = initial_theta;
% damping parameter
lambda = 10;
% choose minibatch for training
batch_size = 10000;
rand_order = randperm(length(training_target));
batch_num = ceil(length(training_target)./batch_size);

% variable for convergence check
costs = zeros(epoches,1);
tolerance = 5e-3;
gap_ratio = 0.1;
min_gap = 5;
max_testgap = max(ceil(gap_ratio*max_iterations),min_gap)+1;
quad_values = zeros(max_testgap,1);
val_his = zeros(max_iterations,1);
%% hessian free
delta = zeros(length(theta),1);
for epoch = 1:epoches
    % minibath is used when computing Bd
    % gradient is computed using the whole training_data
    posi = mod(epoch,batch_num)+1;
    batch_data = training_data(:,rand_order((posi-1)*batch_size+1:posi*batch_size));
    batch_target = training_target(rand_order((posi-1)*batch_size+1:posi*batch_size));
    
    cost0 = dnnCostOnly(theta,batch_data,batch_target,layer_size,0);
    b = -dnnGradOnly(theta,training_data,training_target,layer_size,0);
%     P = computeFDiag(theta,batch_data,batch_target,layer_size);
%     P = P + lambda * ones(size(P));
%     P = P.^0.9;
    P = ones(size(theta));
%% conjugate gradient descent
    r = computeHv(theta,delta,batch_data,batch_target,layer_size,lambda)-b;
    y = r./P;
    p = -y;
    
    delta = delta * 0.9;
    for it =1:max_iterations
        Ap = computeHv(theta,p,batch_data,batch_target,layer_size,lambda);
        alpha = (r'*y)./(p'*Ap);
        fprintf('Alpha: %.4f \n', alpha);
        if alpha < 0
            fprintf('Epoch:%d Iteration:%d, alpha:%.4f \n',epoch,it,alpha);
            fprintf('Program pause \n');
            pause;
            %break;
        end;
        delta = delta + alpha*p;
        belta_old = r'*y;
        r = r + alpha*Ap;
        y = r./P;
        belta = (r'*y)./belta_old;
        p = -y + belta*p;
        
        % convergence check
        quad_value = 0.5*delta'*(-b+r);
        cost = dnnCostOnly(theta+delta,batch_data,batch_target,layer_size,0);
        fprintf('Epoch:%d Iteration:%d Quad cost:%.4f, cost:%.4f \n',epoch,it,quad_value+cost0,cost);
        
        quad_values(mod(it-1,max_testgap)+1) = quad_value;
        val_his(it) = quad_value;
        test_gap = max(ceil(it*gap_ratio),min_gap);
        if it > test_gap
            pre_value = quad_values(mod(it-1-test_gap,max_testgap)+1);
            decrease_rate = (quad_value - pre_value)./quad_value;
            if (decrease_rate < tolerance)
                fprintf('The decrease rate of quad:%.4f \n',decrease_rate);
                fprintf('CG converge. \n');
                break;
            end;
        end;
    end;
%     plot(1:length(val_his),val_his);
%     fprintf('Program pause \n');
%     pause;
    
%% CG end
    theta = theta + delta;
    
    cost1 = dnnCostOnly(delta,batch_data,batch_target,layer_size,0);
    costs(epoch) = cost1;
    
    %Levenberg-Marquardt
    ratio = (cost1 - cost0)./quad_value;
    fprintf('cost1:%.4f cost0:%.4f quad_value:%.4f quad_value+cost0:%.4f \n ',cost1,cost0,quad_value,quad_value + cost0);
    fprintf('The ratios is %.4f \n',ratio);
    if  ratio > 3./4
        lambda = lambda*(2./3);
    elseif ratio < 1./4
        lambda = lambda *(3./2);
    end;
    training_cost = dnnCostOnly(theta,training_data,training_target,layer_size,0);
    fprintf('Training data cost: %.4f \n',training_cost);
    costs(epoch) = training_cost;
end;




end

