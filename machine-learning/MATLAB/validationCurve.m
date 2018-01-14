function [error_train, error_val] = validationCurve(X, y, Xval, yval, lambda_vec)

% return these variables .
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== Real Code ======================

for i = 1 : length(lambda_vec)
	lambda = lambda_vec(i);
	theta = trainLinearReg(X, y, lambda);
	error_train(i) = linearRegCostFunction(X, y, theta, 0);
	error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end


% =============================================================

end
