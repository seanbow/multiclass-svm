function [y, fval] = predict_sc_svm(X, SVM)

% f(x) = sum_i SVM.a(:,i) * kernel(sv(i), X)..
%
% then project f(x) onto code vectors

Kx = SVM.kernel(SVM.normalize(X), SVM.svs, SVM.hyperparams);

a_times_code = SVM.a' * SVM.code;
projected = Kx * a_times_code;
[fval, y] = max(projected, [], 2);
fval = fval';
y = y';

% y = zeros(1, size(X,1));
% fval = zeros(1, size(X,1));
% for i = 1:size(X,1)
% %     f = SVM.a * Kx(i,:)';
%     
% %     projected = f'*SVM.code;
%     projected = Kx(i,:) * a_times_code;
%     
%     [fval(i), y(i)] = max(projected);
% end