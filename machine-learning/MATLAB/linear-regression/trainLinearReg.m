% 训练线性回归
function [theta, J] = trainLinearReg(X, y, lambda, iter)

% Initialize Theta
initial_theta = zeros(size(X, 2), 1); 

% Create "short hand" for the cost function to be minimized
costFunction = @(t) linearRegCostFunction(X, y, t, lambda);

% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', iter, 'GradObj', 'on');

% Minimize using fmincg
[theta, J] = fmincg(costFunction, initial_theta, options);

end
