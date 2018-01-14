% Initialization
clear ; close all; clc

% Load Training Data
fprintf('Loading Data ...\n')

% load ('ex5data1.mat');

fprintf('Load data is completed. Press enter to continue.\n');
pause;

%% =========== Part 0: 预备区域 =============

% todo: X添加转置单元
% m = Number of examples
m = size(X, 1);

%% =========== Part 1: 线性回归区域 =============

% 做均值归一化处理
[X_poly, mu, sigma] = featureNormalize(X);
% Add Ones
X_poly = [ones(m, 1), X_poly];      

% 预测值做均值归一化处理
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
% Add Ones 
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];  


% 执行线性回归训练
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);                   

%% =========== Part 2: 逻辑回归区域 =============


% 值预测
result = sigmoid(X * theta);
p = (result >= 0.5);
fprintf('Logistic Train Accuracy: %f\n', mean(double(p == y)) * 100);

% 多类别分类处理
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);
% 多分类预测
result = sigmoid(X * all_theta'); % result是m x k维的矩阵
% 找出让h(i)最大的i，就是预测的类别
% 返回每行最大值，结果存在ans里，index里存的是每行最大值的列位置。
[ans, index] = max(result, [], 2);
% 预测值
p = index;
% 准确率
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% =========== Part 3: 神经网络区域 =============

% 初始化Θ值
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters 
nn_params = [initial_Theta1(:) ; Thetainitial_Theta22(:)];

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% 执行梯度下降
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% 值预测
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);
fprintf('\nNeural NetWork Training Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% =========== Part 4: 绘制图谱区域 =============

% J(Θ)与迭代次数关系图
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% 学习曲线:learningCurve.m
lambda = 0;
[error_train, error_val] = learningCurve([ones(m, 1) X], y, ...
                  						[ones(size(Xval, 1), 1) Xval], yval, ...
                  						lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

% 交叉验证曲线(λ):validationCurve.m
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
[error_train, error_val] = validationCurve(X, y, Xval, yval, lambda_vec);

plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
