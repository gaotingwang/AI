% 簇分配
function idx = findClosestCentroids(X, centroids)

% Set K
K = size(centroids, 1);

% return the following variables correctly.
idx = zeros(size(X,1), 1);

% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%

% 进行簇分配，相当于给每一个样本打上标签进行分类，看每一个样本所属的最近的k

for i = 1 : size(X,1)
	temp = zeros(K,1); % 保存样本与聚类中心之间的距离
	for j = 1 : K
		length = X(i, :) - centroids(j, :);		
		temp(j) = length * length';
	end
	[ans index] = min(temp); % 找出最小距离的index
	idx(i) = index;
end

% =============================================================

end

