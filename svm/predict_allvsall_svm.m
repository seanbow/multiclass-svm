function [classid, fvals] = predict_allvsall_svm(X, SVM)
% Returns the class prediction from a all-vs-all svm 'SVM' for a data point
% X or a vector of data points X
%
% X is an m by N matrix of m examples in N-dimensional space

% Run x through all binary SVMs.
% Let f_{ij} be the classifier where class i were positive examples and
% class j negative. Then the result of the all-vs-all classifier is
%  k = arg max_i sum_j f_ij(x)

% let vals(:,:,k) = f_{ij}(X(k,:))

fvals = zeros(SVM.K, SVM.K, size(X, 1));
votes = zeros(size(X,1), SVM.K);

Xn = SVM.normalize(X);

for i=1:SVM.K-1
    for j = i+1:SVM.K
        prediction = SVM.binary_svms{i}{j}.predict(Xn);
        fvals(i, j, :) = prediction;
        votes(prediction >= 0, i) = votes(prediction >= 0, i) + 1;
        votes(prediction < 0, j) = votes(prediction < 0, j) + 1;
    end
end

% Note that f_{ij} = -f_{ji}, and as we only filled in the upper triangular
% portion, we reconstruct the full fvals as:
fvals = fvals - permute(fvals, [2 1 3]);

% Now get the actual predictions
fvals = sum(fvals, 2);
fvals = reshape(fvals, SVM.K, []);
fvals = fvals';

% [fvals, classid] = max(fvals', [], 1);


% follow sklearn. scale confidences (fvals) to (0.5, 0.5)
scale = (0.5 - eps) ./ max([abs(max(fvals)), abs(min(fvals))]);
scaled_confidence  = scale * fvals;

[fvals, classid] = max(votes + scaled_confidence, [], 2);
classid = classid';