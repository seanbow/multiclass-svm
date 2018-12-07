addpath svm

% Generate data


Nc = 500; % data points *per class* 
d = 2;   % dimension
K = 3;   % number of classes

X = []; Y = [];

% Set mean of class k to 2*k*ones(N,1)
for k = 1:K
    mu = 2*k*ones(1,d);
    S = randn(d);
    S = S*S';
%     S = S/det(S);
    S = d*S/trace(S);
    X = [X ; mvnrnd(mu, S, Nc)];
    Y = [Y k*ones(1,Nc)];
end


% Randomly sample means
% scale = 10;
% for k = 1:K
%     meank = unifrnd(-scale, scale, 1, N);
%     X = [X ; mvnrnd(meank, eye(N), m)];
%     Y = [Y k*ones(1,m)];
% end
    

%% One vs All
% SVM = get_model("OVA", K);
% SVM.kernel = 'rbf';
% SVM.hyperparams.C = 1e-1;
% SVM.hyperparams.gamma = 1;

%% All vs All
SVM = get_model("AVA", K);
SVM.kernel = 'rbf';
SVM.hyperparams.C = 1e-2;
SVM.hyperparams.gamma = 10;

%% LLW
% SVM = get_model("LLW", K);
% SVM.hyperparams.C = 0.1;

%% CS
% SVM = get_model("CS", K);
% SVM.hyperparams.gamma = 1;
% SVM.hyperparams.C = 1e-2;

%% SC / Simplex Cone
% SVM = get_model("SC", K);
% SVM.hyperparams.C = 1e-3;

%% SH / Simplex Halfspace
% SVM = get_model("SH", K);
% SVM.hyperparams.C = 1e-5;
% SVM.kernel = "polynomial";
% SVM.hyperparams.d = 9;
% 
outer_tstart = tic;
SVM = train_model(SVM, X, Y, 'max_iters', 500, 'verbose', 3, 'batch_size', 200, 'tol', 1e-4, "batch_method", "probabilistic");
toc(outer_tstart)
plot_multiclass_model(SVM.trained, X, Y);

% crossvalidate(SVM,X,Y,3,'tol',1e-3,'batch_size',500)

if 0
    param_range.C = 10.^(-1:5);
    
    result = optimize_hyperparameters(SVM, X, Y, param_range)
end