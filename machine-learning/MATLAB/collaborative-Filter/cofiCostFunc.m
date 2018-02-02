function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% J
predict = X * Theta'; % 维度num_movies x num_users
error2 = (predict - Y) .* (predict - Y);
J = sum(sum(R .* error2)) / 2; % R(i,j)不为0之和 

% 正则化J
J += (lambda / 2) * sum(sum(Theta .* Theta));
J += (lambda / 2) * sum(sum(X .* X));


% X_grad
for i = 1 : size(X, 1) % 遍历每一个电影
	idx = find(R(i, :) == 1); % 每部电影沿用户方向查找有用户评价的位置
	Theta_temp = Theta(idx, :); % 取R(i, :)中为1的用户对应的theta
	Y_temp =Y(i, idx) % 取R(i, :)中为1的Y值
	X_grad(i, :) = (X(i, :) * Theta_temp' - Y_temp) * Theta_temp;
	% 正则化
	X_grad(i, :) += (lambda * X(i, :));
end

% Theta_grad
for j = 1 : size(Theta, 1) % 遍历每一个用户
	idx = find(R(:, j) == 1); % 每个用户沿电影方向查找有用户评价的位置
	X_temp = X(idx, :);
	Y_temp =Y(idx, j) % 取R(i, :)中为1的Y值
	Theta_grad(j, :) = (X_temp * Theta(j, :)' - Y_temp)' * X_temp;
	% 正则化
	Theta_grad(j, :) += (lambda * Theta(j, :)); 
end


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
