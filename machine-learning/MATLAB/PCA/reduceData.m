% 数据降维
function Z = reduceData(X, U, K)


% return the following variables correctly.
Z = zeros(size(X, 1), K);

% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

U_reduce = U(:, 1:K);

Z = X * U_reduce;

% =============================================================

end
