% Initialization
clear ; close all; clc
addpath('../');

%% =========== Part 1: 预备区域 =============

fprintf('Loading Data ...\n')

load ('mydata6.mat');

fprintf('Load data is completed. Press enter to continue.\n');
pause;

%% =========== Part 2: PCA区域 =============

% 首先进行均值归一化
[X_norm, mu, sigma] = featureNormalize(X);

% 指定PCA算法
[U, S] = pca(X_norm);

%% ================== end ========================




