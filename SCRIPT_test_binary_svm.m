addpath svm

% Generate data
m = 5000;
N = 2;

X1 = mvnrnd(3*ones(1,N), eye(N), m);
X2 = mvnrnd(zeros(1,N), eye(N), m);

Y1 = ones(1,m);
Y2 = -ones(1,m);

X = [X1 ; X2]; Y = [Y1 Y2];

tic;

hyperparams.gamma = 1;
hyperparams.C = 0.1;

SVM = train_binary_svm(X,Y, 'rbf', hyperparams, 'verbose', 2, 'batch_size', 500);

toc

% tic; smo_mex(X, Y, hyperparams.C, hyperparams.gamma); toc

plot_binary_svm(SVM, X, Y);
