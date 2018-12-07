function [classid, fval] = predict_cs_svm(X, SVM)

% for a single data point x and class k the classifier function is given by 
%  sum_i tau(i,k) kernel(x, sv_i)

Kx = SVM.kernel(SVM.normalize(X), SVM.svs, SVM.hyperparams);

fvals = zeros(size(X,1), SVM.K);

for i = 1:size(X,1)
    for k = 1:SVM.K
        fvals(i,k) = SVM.tau(:, k)' * (Kx(i, :) + 1)';
    end
end

[fval, classid] = max(fvals, [], 2);