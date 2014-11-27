function [ error ] = gradientCheck( theta,funObj)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
epsilion = 1e-4;
m = length(theta);
[~,grad] = funObj(theta);
grad_2 = zeros(m,1);
for i = 1:m
    delta = zeros(m,1);
    delta(i) = epsilion;
    [cost_1,~] = funObj(theta + delta);
    [cost_2,~]= funObj(theta - delta);
    grad_2(i) = (cost_1-cost_2)./epsilion./2;
end;

%display the error
disp([grad grad_2 (grad-grad_2)]);
error = norm(grad - grad_2);

end

