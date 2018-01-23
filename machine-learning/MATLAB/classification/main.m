% Initialization
clear ; close all; clc
addpath('../');

%% =========== Part 1: 预备区域 =============

fprintf('Loading Data ...\n')

% 加载数据，必要时按照3:1:1的比例拆分出：训练集、交叉验证集、测试集
load ('mydata2.mat');

% % 加载excel中的数据
% pkg install D:\Octave\Octave-3.8.2\src\io-2.2.6.tar.gz; % 只需运行一次，不需要每次运行
% pkg load io;
% [Train, txt]= xlsread('a4.xlsx', 'A1:J19202');

Train = [X, y];
rand_indices = randperm(size(Train, 1)); %返回一个从1-m的包含m个数的随机排列,每个数字只出现一次

X_train = Train(rand_indices(1:3000), 1 : (size(Train, 2) - 1));
y_train = Train(rand_indices(1:3000), size(Train, 2));
train_size = size(X_train, 1); % Number of train

X_val = Train(rand_indices(3001:4000), 1 : (size(Train, 2) - 1));
y_val = Train(rand_indices(3001:4000), size(Train, 2));
val_size = size(X_val, 1); % Number of validate

X_test = Train(rand_indices(4001:5000), 1 : (size(Train, 2) - 1));
y_test = Train(rand_indices(4001:5000), size(Train, 2));
test_size = size(X_test, 1); % Number of test

num_labels = 10; % 多分类，当前类别数

fprintf('Load data is completed. Press enter to continue.\n');
pause;

%% =========== Part 2: 逻辑回归区域 =============

% 多分类处理
feature_num = size(X, 2); % 特征数
init_theta = zeros(num_labels, feature_num + 1); % 初始化theta

X_train = [ones(train_size, 1) X_train];
lambda = 4;
options = optimset('GradObj', 'on', 'MaxIter', 40);

for i = 1:num_labels
	thetai = init_theta(i, :);
	% 这里特别处理是(y_train == i)表示第i个训练实例是否属于类k, 计算针对第k类的theta
	[costTheta] = fmincg (@(t)(logisticRegCostFunction(t, X_train, (y_train == i), lambda)), thetai(:), options);
	init_theta(i, :) = costTheta';
end

% 准确率
X_train = X_train(:, 2:end);
pred = predictOneVsAll(init_theta, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);
pred = predictOneVsAll(init_theta, X_val);
fprintf('\nValidation Set Accuracy: %f\n', mean(double(pred == y_val)) * 100);
pred = predictOneVsAll(init_theta, X_test);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

%% =============== Part 3: SVM区域 ==================

% 使用SVM
fprintf('Prepare to use SVM. Press enter to continue.\n');
pause;
load('mydata3.mat');

% 线性核函数
% C = 1; 
% model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);

% 遍历list寻找使误差最小的C和sigma
% list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% maxError = 1; % 假设模型完全不匹配，误差率为1
% for i = 1 : length(list)
% 	for j = 1 : length(list)
% 		C_train = list(i);
% 		sigma_train = list(j);
% 		model= svmTrain(X, y, C_train, @(x1, x2) gaussianKernel(x1, x2, sigma_train));
% 		predictions = svmPredict(model, Xval);
% 		errors = mean(double(predictions ~= yval));
% 		% 取误差最小的C和sigma
% 		if errors <= maxError
% 			C = C_train;
% 			sigma = sigma_train;
% 			maxError = errors;
% 		end
% 	end
% end

% SVM Parameters
C = 1; sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
model

fprintf('Program paused. Press enter to continue.\n');
pause;

% 准确率
% svmPredict该函数主要思路是先计算出p(theta' * X)，根据p大于0或小于0决定在决策边界的哪一边
% svm的假设函数：p >= 0点的预测值为1，< 0 预测值为0
pred = svmPredict(model, X);
fprintf('Test Accuracy: %f\n', mean(double(pred == y)) * 100);

%% =============== Part 4: 使用LIBSVM区域 ==================

% 数据需要为double型的
% double(Train);

addpath('./libsvm');

% libsvm使用
C = 30;
sigma = 1;
opt = sprintf('%s %f %s %f %s', '-c', C, '-g', sigma, '-h 0');
% 模型训练
model = libsvmtrain(y_train, X_train, opt);
% 模型准确率预测
[predict_label, accuracy, dec_values] = libsvmpredict(y_train, X_train, model);
[predict_label, accuracy, dec_values] = libsvmpredict(y_val, X_val, model);
[predict_label, accuracy, dec_values] = libsvmpredict(y_test, X_test, model);



%% 将多分类转换为二分类，如何获取代价函数J的值? 怎么绘制学习曲线??





%% ================== end ========================




