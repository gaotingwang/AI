% Initialization
clear ; close all; clc
addpath('../');

%% =========== Part 1: 预备区域 =============

fprintf('Loading Data ...\n')

% 加载数据，必要时按照3:1:1的比例拆分出：训练集、交叉验证集、测试集
load ('mydata4.mat');

Train = [X, y];
rand_indices = randperm(size(Train, 1)); %返回一个从1-m的包含m个数的随机排列,每个数字只出现一次

% 训练集
X_train = Train(rand_indices(1:3000), 1 : (size(Train, 2) - 1));
y_train = Train(rand_indices(1:3000), size(Train, 2));
train_size = size(X_train, 1); % Number of train

% 验证集
X_val = Train(rand_indices(3001:4000), 1 : (size(Train, 2) - 1));
y_val = Train(rand_indices(3001:4000), size(Train, 2));
val_size = size(X_val, 1); % Number of validate

% 测试集
X_test = Train(rand_indices(4001:5000), 1 : (size(Train, 2) - 1));
y_test = Train(rand_indices(4001:5000), size(Train, 2));
test_size = size(X_test, 1); % Number of test

input_layer_size  = 400;  % 基础特征数
hidden_layer_size = 25;   % 隐藏层的隐藏单元数
num_labels = 10; % 多分类，当前类别数

fprintf('Load data is completed. Press enter to continue.\n');
pause;

%% =========== Part 1: 神经网络区域 =============

lambda = 1;

% 1. 初始化Θ值
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters 
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% 在costFunction处理：
% 2. 根据前向传播算法计算h(x)
% 3. 计算J(Theta)
% 4. 利用反向算法计算J(Theta)对Theta的偏导
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);
% 5. 梯度检查
% [cost, grad] = costFunction(nn_params);
% numgrad = computeNumericalGradient(costFunction, nn_params);
% disp([numgrad grad]);

% 6. 执行梯度下降
options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(costFunction, nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% 值预测
h1 = sigmoid([ones(test_size, 1) X_test] * Theta1');
h2 = sigmoid([ones(test_size, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);
fprintf('\nNeural NetWork Training Set Accuracy: %f\n', mean(double(p == y_test)) * 100);


%% =========== Part 2: 绘制图谱区域 =============

% 1. 绘制收敛图 MAC run: setenv("GNUTERM","qt")
figure;
plot(1:numel(cost), cost, '-g', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

fprintf('Plot the convergence graph is completed. Press enter to continue.\n');
pause; 

% 2. 学习曲线:learningCurve.m
error_train = zeros(train_size, 1); % 训练集误差
error_val   = zeros(train_size, 1); % 交叉验证集误差
for i = 1:150
	learnX = X_train(1:i, :);
	learny = y_train(1:i);
	costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, learnX, learny, lambda);
	[theta] = fmincg(costFunction, nn_params, options); % 样本数为i时，通过高级梯度下降计算出theta
	error_train(i) = nnCostFunction(theta, input_layer_size, hidden_layer_size, num_labels, learnX, learny, 0); % lambda传入0计算训练误差J_train
	error_val(i) = nnCostFunction(theta, input_layer_size, hidden_layer_size, num_labels, X_val, y_val, 0); % lambda传入0计算交叉验证误差J_cv
end

figure;
plot(1:train_size, error_train, 1:train_size, error_val);
title('Learning curve for linear regression');
legend('Train', 'Cross Validation');
xlabel('Number of training examples');
ylabel('Error');
axis([0 150 0 15]);

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:train_size
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Plot the Learning Curve is completed. Press enter to continue.\n');
pause; 

% 3. 交叉验证曲线(λ):validationCurve.m
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
for i = 1 : length(lambda_vec)
	lambdai = lambda_vec(i);
	costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambdai);
	[theta] = fmincg(costFunction, nn_params, options);
	error_train(i) = nnCostFunction(theta, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, 0);
	error_val(i) = nnCostFunction(theta, input_layer_size, hidden_layer_size, num_labels, X_val, y_val, 0);
end

figure;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', lambda_vec(i), error_train(i), error_val(i));
end


