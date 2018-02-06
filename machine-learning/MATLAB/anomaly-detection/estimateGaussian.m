function [mu sigma2] = estimateGaussian(X)

% Useful variables
[m, n] = size(X);

% return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);


% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

mu = mean(X);

% 将均值复制多行
% mus = mu(ones(m,1),:);
mus = repmat(mu,m,1);

sigma2 = mean((X - mus) .* (X - mus));


% =============================================================


end
