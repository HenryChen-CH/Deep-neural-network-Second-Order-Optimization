function [ theta, cost ] = hessianFree( funObj,initial_theta, options )
%UNTITLED8 Summary of this function goes here
%   This function implement the hessian free optimization
% [cost, grad] = funObj(theta)
% reference : http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/
%% initialize the parameters
epoches = options.iterations;
theta = initial_theta;
% damping parameter
lambda = 1;


%% hessian free
for epoch = 1:epoches
    [cost,b] = funObj(theta);
%% conjugate gradient descent
    delta_x = randn(length(theta),1) * 0.1;
    
    grad_0 = (computeHV(funObj,theta,delta_x, lambda) + b);
    d = -grad_0;
    for it = 1 : 4
        grad = computeHV(funObj,theta,delta_x,lambda) + b;
        d_hessian = computeHV(funObj,theta,d,lambda);
        alpha = -d'*grad./(d'* d_hessian);
        delta_x(:) = delta_x(:) + alpha * d;
        
        %compue the direction of the next step
        grad_1 = computeHV(funObj,theta,delta_x,lambda) + b;
        aa = grad_1' * d_hessian;
        bb = d' * d_hessian;
        beta = grad_1' * d_hessian./(d' * d_hessian);
        d(:) = - grad_1(:) + beta * d(:);
    end;
    theta(:) = theta(:) + delta_x(:);
    
    fprintf('Epoch : %d Cost function : %.4f \n',epoch, cost);
end;


end

function [y] = computeHV(funObj, theta ,x, lambda)

epsilon = 1e-4;

[~,grad_1] = funObj(theta);
[~,grad_2] = funObj(theta + epsilon * x);
y = (grad_2 - grad_1)./epsilon;
y = y + lambda * x;
end


