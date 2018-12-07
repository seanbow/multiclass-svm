function inds = crossvalidation_indices(X, nfolds)
% Returns a set of data indices to perform k-fold cross validation with a
% given dataset X.
%
% Splits the data into nfolds folds such that inds(i) == k if data point i
% is a member of the kth fold.

fold_size = ceil(size(X,1) / nfolds);

inds = repmat(1:nfolds, 1, fold_size);
inds = inds(randperm(numel(inds)));
inds = inds(1:size(X,1));