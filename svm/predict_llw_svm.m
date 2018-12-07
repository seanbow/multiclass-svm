function [classid, fvals] = predict_llw_svm(X, SVM)

% a single data point x corresponds to a vector response f = [f1 f2 ... fk]
% for each class K, where
%    f_j(x) = b_j + sum_i c_{ij} K(x_i, x)

fvals = zeros(size(X,1), SVM.K);

Xn = SVM.normalize(X);

% cache kernel evaluations
% Kxs = SVM.kernel(Xn, SVM.svs, SVM.hyperparams);

for i = 1:size(Xn, 1)
    Kix = SVM.kernel(Xn(i,:), SVM.svs, SVM.hyperparams);
    for j = 1:SVM.K
%         fvals(i,j) = SVM.b(j) + SVM.c(:,j)'*Kxs(i,:)';
        fvals(i,j) = SVM.b(j) + SVM.c(:,j)'*Kix';
    end
end

[fvals, classid] = max(fvals, [], 2);