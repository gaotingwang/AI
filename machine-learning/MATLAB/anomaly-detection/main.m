% Initialization
clear ; close all; clc
addpath('../');

%% =========== Part 1: 预备区域 =============

fprintf('Loading Data ...\n')

load ('mydata7.mat');

fprintf('Load data is completed. Press enter to continue.\n');
pause;

%% =========== Part 2: 异常检测区域 =============

%  Apply the same steps to the larger dataset
[mu sigma2] = estimateGaussian(X);

%  Training set 
p = multivariateGaussian(X, mu, sigma2);

%  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2);

%  Find the best threshold
[epsilon F1] = selectThreshold(yval, pval);

%% ================== end ========================




