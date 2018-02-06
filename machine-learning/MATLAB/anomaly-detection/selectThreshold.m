% 通过查准率和召回率选取合适的epsilon
function [bestEpsilon bestF1] = selectThreshold(yval, pval)

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    % 验证样本标签值
    cvPredictions = double(pval < epsilon);

    % 真阳性
    tp = sum((cvPredictions == 1) & (yval == 1));
    % 假阳性
    fp = sum((cvPredictions == 1) & (yval == 0));
    % 假阴性
    fn = sum((cvPredictions == 0) & (yval == 1));
    % 查准率
    prec = tp / (tp + fp);
    % 召回率
    rec = tp / (tp + fn);

    % F1Score
    F1 = (2 * prec * rec) / (prec + rec);


    % =============================================================
    % 取使F1Score最大的epsilon
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
