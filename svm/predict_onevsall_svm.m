function [classid, fval] = predict_onevsall_svm(X, SVM)
% Returns the class prediction from a one-vs-all svm 'SVM' for a data point
% X or a vector of data points X
%
% X is an m by N matrix of m examples in N-dimensional space

% Run x through all binary SVMs and return the maximum classifier value
vals = zeros(size(X, 1), SVM.K);

for i=1:SVM.K
    vals(:, i) = SVM.binary_svms{i}.predict(SVM.normalize(X));
end

[fval, classid] = max(vals, [], 2);