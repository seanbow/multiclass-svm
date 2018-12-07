function SVM = train_onevsall_svm(X, Y, K, kernel_type, hyperparams, varargin)
% X is a matrix of training data, m by N
%   -- (m examples of dimension N, one example per row)
%
% Y is an m-dimensional vector of labels in {1, 2, \dots, K}

[SVM, Xn] = svm_preprocess(X, 'normalize', true);

% We need to build K different binary classifiers, the ith of which will be
% trained on the ith class vs all others
SVMs = {};
for class_id = 1:K
    Xi = Xn(Y == class_id, :);
    Yi = ones(1, size(Xi, 1));
    
    Xrest = Xn(Y ~= class_id, :);
    Yrest = -1 * ones(1, size(Xrest, 1));
    
    SVMs{class_id} = train_binary_svm([Xi;Xrest], [Yi Yrest], kernel_type, hyperparams, 'normalize', false, varargin{:});
end

SVM.type = "OVA";
SVM.binary_svms = SVMs;
SVM.K = K;
SVM.kernel = kernel_type;
SVM.hyperparams = hyperparams;

SVM.predict = @(X) predict_onevsall_svm(X, SVM);

