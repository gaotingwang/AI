% Initialization
clear ; close all; clc
addpath('../');

%% =========== Part 1: 预备区域 =============

fprintf('Loading Data ...\n')

load ('mydata5.mat');

fprintf('Load data is completed. Press enter to continue.\n');
pause;

%% =========== Part 2: K-Means区域 =============

K = 3;
initial_centroids = [3 3; 6 2; 8 5];
max_iters = 10;

% % 簇分配
% idx = findClosestCentroids(X, initial_centroids);

% % 移动聚类中心
% centroids = computeCentroids(X, idx, K);

% Run K-Means algorithm. The 'true' at the end tells our function to plot
% the progress of K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== end ========================


