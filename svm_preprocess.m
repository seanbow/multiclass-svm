function [SVM, Xn] = svm_preprocess(X, varargin)

inputExist = find(strcmpi(varargin, 'normalize'));
if inputExist
    do_normalize = varargin{inputExist + 1};
else
    do_normalize = true;
end

if do_normalize
    % "soft" normalization
%     SVM.shift = mean(X,1);
%     SVM.scale = std(X,0,1);
    
    % normalize to [-1, 1]
%     SVM.shift = mean(X,1);
%     SVM.scale = max(abs(bsxfun(@minus, X, SVM.shift)), [], 1);

    % normalize to [0, 1]
    SVM.shift = min(X, [], 1);
    SVM.scale = max(X, [], 1) - SVM.shift;

    SVM.scale = max(SVM.scale, 2*eps);
    SVM.normalize = @(X) bsxfun(@rdivide, bsxfun(@minus, X, SVM.shift), SVM.scale);
    SVM.unnormalize = @(X) bsxfun(@plus, bsxfun(@times, X, SVM.scale), SVM.shift);
else
    SVM.shift = zeros(1, size(X,2));
    SVM.scale = ones(1, size(X,2));
    SVM.normalize = @(X) X;
    SVM.unnormalize = @(X) X;
end

Xn = SVM.normalize(X);