% Initialization
clear ; close all; clc
addpath('../');

%% =========== Part 1: 预备区域 =============

fprintf('Loading Data ...\n')

% 1. 加载数据，必要时按照3:1:1的比例拆分出：训练集、交叉验证集、测试集
load ('mydata1.mat');
% Number of examples
m = size(X_poly, 1);

% 2. 做均值归一化处理
[X_poly, mu, sigma] = featureNormalize(X_poly);
% Add Ones
X_poly = [ones(m, 1), X_poly];

% 验证集、测试集做均值归一化处理
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test]; % Add Ones 

X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val]; % Add Ones 

fprintf('Load data is completed. Press enter to continue.\n');
pause;

%% =========== Part 2: 线性回归区域 =============

% 执行线性回归训练，根据绘图区的结果进行分析，来调整lambda, m, 特增值
lambda = 3;
num_iters = 200;
[theta, J_history] = trainLinearReg(X_poly, y, lambda, num_iters);

% 准确率
fprintf('\nTraining Set Accuracy: %f\n', mean(double(X_poly_test * theta == ytest)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: 绘制图谱区域 =============

% 绘制收敛图 MAC run: setenv("GNUTERM","qt")
figure;
plot(1:numel(J_history), J_history, '-g', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

fprintf('Plot the convergence graph is completed. Press enter to continue.\n');
pause; 

% 学习曲线:learningCurve.m
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda, num_iters);

figure;
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression');
legend('Train', 'Cross Validation');
xlabel('Number of training examples');
ylabel('Error');
axis([0 13 0 150]);

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Plot the Learning Curve is completed. Press enter to continue.\n');
pause; 

% 交叉验证曲线(λ):validationCurve.m
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
[error_train, error_val] = validationCurve(X_poly, y, X_poly_val, yval, num_iters, lambda_vec);

figure;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

%% ================== end ========================



