% Initialization
clear ; close all; clc
addpath('../');

%% =========== Part 1: 预备区域 =============

fprintf('Visualizing example dataset for PCA.\n\n');

load ('mydata6.mat');

%  Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bo');
axis([0.5 6.5 2 8]); axis square;

fprintf('Load data is completed. Press enter to continue.\n');
pause;

%% =========== Part 2: PCA区域 =============

% 首先进行均值归一化
[X_norm, mu, sigma] = featureNormalize(X);

% PCA算法
[U, S] = pca(X_norm);

fprintf('\nDimension reduction on example dataset.\n\n');

% Plot the normalized dataset (returned from pca)
plot(X_norm(:, 1), X_norm(:, 2), 'bo');
axis([-4 3 -4 3]); axis square

% 将数据降至k = 1维
K = 1;
Z = reduceData(X_norm, U, K);

% 压缩数据还原
X_rec  = recoverData(Z, U, K);


%  Draw lines connecting the projected points to the original points
hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
hold off

fprintf('Program finished.\n');


%% ================== end ========================




