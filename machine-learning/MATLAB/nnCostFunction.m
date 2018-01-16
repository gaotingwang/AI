% 神经网络代价函数
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% return the following variables
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

% ========================== Part 1 ==========================
% 	求代价函数J：利用向前传播算法，先算出h(x),再根据h(x)和y来求J.

% 1. 先根据前向传播算出h(x)的值
%    先给X添加偏置单元1
X = [ones(m,1) X];

%    算出隐藏层的计算结果
result1 = sigmoid(X * Theta1');
result1 = [ones(m, 1) result1]; % 给隐藏层加偏置单元1

%    计算输出层的结果
result2 = sigmoid(result1 * Theta2');
result = result2;

% 2. 根据预测出的结果h(x)来求代价函数，此处得出结果还不包括则正则化
for k = 1:num_labels
	yk = (y == k);
	resultk = result(:,k); % 标签k的预测结果h(x)
	Jk = (-1 / m) * (yk' * log(resultk) + (1 - yk)' * log(1-resultk));
	J = J + Jk;
end

% ========================== Part 2 ==========================
%		使用反向传播算法，计算每一层的梯度Theta1_grad and Theta2_grad

for i = 1:m
	% 设定a(1)，也就是输入层的激励函数：a_1=x(i)
	a_1 = X(i,:); % 已经添加过偏置单元了，直接取
	% 执行向前传播，分别计算出第2层和第3层的激励值(z_2,a_2,z_3,a_3)
	z_2 = a_1 * Theta1';
	a_2 = [1 sigmoid(z_2)]; % 计算前添加偏置单元
	z_3 = a_2 * Theta2';
	a_3 = sigmoid(z_3);
	% 计算每一层的误差
	for k = 1:num_labels
		y_i = zeros(num_labels, 1);
		y_i(y(i)) = 1; % 输出值y变成对应k值为1的形式 [0,0,1,0,0]
		delta_3 = a_3' - y_i;
	end
	% 计算隐藏层的delta不应该包括偏置单元
	delta_2 = (Theta2(: , 2:end))' * delta_3 .* sigmoidGradient(z_2');
	% delta_2 = delta_2(2:end); % 舍弃偏置单元对应的delta
	% 计算每一层的梯度
	Theta1_grad += (1 / m) * (delta_2 * a_1);
	Theta2_grad += (1 / m) * (delta_3 * a_2);
end

% ========================== Part 3 ==========================
%		对代价函数和梯度进行正则化
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% 1. 正则化代价函数
%    去除权重里的偏置单元对应的参数，它们不需要参加正则化
noBaisTheta1 = Theta1(: , 2:end);
noBaisTheta2 = Theta2(: , 2:end);

J = J + (lambda / (2 * m)) * (sum(sum(noBaisTheta1 .* noBaisTheta1)) + sum(sum(noBaisTheta2 .* noBaisTheta2)));

% 2. 正则化梯度
%    每一层的第一列不参与正则化
Theta1_nobais = [zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_nobais = [zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
Theta1_grad = Theta1_grad + (lambda / m) * Theta1_nobais;
Theta2_grad = Theta2_grad + (lambda / m) * Theta2_nobais;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
