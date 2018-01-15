% 线性回归代价函数计算
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y); % number of training examples

% init the value to be returned
J = 0;
grad = zeros(size(theta));

% ====================== Real Code ======================

% theta_0 不参与正则化
newTheta = theta(2:length(theta));
J = (1 / (2 * m)) * ((X * theta - y)' * (X * theta - y)) + (lambda / (2 * m)) * (newTheta' * newTheta);

grad = (1 / m) * X' * (X * theta - y) + (lambda / m) * theta;
% theta0 不参与正则化,所以需要重新计算grad(1)
grad(1) = (1 / m) * X(:,1)' * (X * theta - y);


% ========================================================
grad = grad(:);

end
