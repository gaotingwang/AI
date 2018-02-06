% Initialization
clear ; close all; clc
addpath('../');

%% =========== Part 1: 预备区域 =============

fprintf('Loading Data ...\n')

load ('mydata7.mat');

fprintf('Load data is completed. Press enter to continue.\n');
pause;

%% =========== Part 2: 异常检测区域 =============

% 计算高斯分布的mu和sigma^2
[mu sigma2] = estimateGaussian(X);

% 可以通过hist函数画图，查看每个维度是否符合正太分布
% X_i = X(:,i);
% hist(X_i, 50);

% 计算训练样本的概率
p = multivariateGaussian(X, mu, sigma2);

% 计算验证样本的概率
pval = multivariateGaussian(Xval, mu, sigma2);

% 异常预测属于偏斜类，通过查准率和召唤率选取合适的epsilon
[epsilon F1] = selectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 1.38e-18)\n');
fprintf('   (you should see a Best F1 value of 0.615385)\n');
fprintf('# Outliers found: %d\n\n', sum(p < epsilon));


%% ================== end ========================




