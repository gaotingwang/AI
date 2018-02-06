% 移动聚类中心
function centroids = computeCentroids(X, idx, K)

% Useful variables
[m n] = size(X);

% return the following variables correctly.
centroids = zeros(K, n);

% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%

% 移动聚类中心
tagX = [X idx];
for k = 1 : K
	% 过滤出tagX中标记为k的样本
	index = tagX(:,size(tagX,2)) == k;
	simple = tagX(index, 1:size(tagX, 2) -1);
	% 求平均值作为下一次聚类中心
	centroids(k,:) = mean(simple);
end


% =============================================================

end

