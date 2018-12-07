function SVM = train_allvsall_svm(X, Y, K, kernel, hyperparams, varargin)
% X is a matrix of training data, m by N
%   -- (m examples of dimension N, one example per row)
%
% Y is an m-dimensional vector of labels in {1, 2, \dots, K}

[SVM, Xn] = svm_preprocess(X, 'normalize', true);

% We need to build K-1 different binary classifiers for *each* input class.
% Each class k will be trained against each class j != k
SVMs = {};
for k = 1:K
    SVMs{k} = {};
    for j = k+1:K
        Xk = Xn(Y == k, :);
        Yk = ones(1, size(Xk, 1));

        Xj = Xn(Y == j, :);
        Yj = -1 * ones(1, size(Xj, 1));

        SVMs{k}{j} = train_binary_svm([Xk;Xj], [Yk Yj], kernel, hyperparams, 'normalize', false, varargin{:});
    end
end

SVM.type = "AVA";
SVM.binary_svms = SVMs;
SVM.K = K;
SVM.kernel = kernel;
SVM.hyperparams = hyperparams;

SVM.predict = @(X) predict_allvsall_svm(X, SVM);

