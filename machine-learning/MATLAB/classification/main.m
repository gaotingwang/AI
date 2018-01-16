% Initialization
clear ; close all; clc
addpath('../');

%% =========== Part 1: 预备区域 =============

fprintf('Loading Data ...\n')

% 加载数据，必要时按照3:1:1的比例拆分出：训练集、交叉验证集、测试集
load ('mydata2.mat');

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
options = optimset('GradObj', 'on', 'MaxIter', 200);

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


%% 将多分类转换为二分类，如何获取代价函数J的值 ???

%% =========== Part 2: SVM区域 =============

% 使用SVM
fprintf('Prepare to use SVM. Press enter to continue.\n');
pause;


%% ================== end ========================




