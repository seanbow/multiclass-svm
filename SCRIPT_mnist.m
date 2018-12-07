addpath ../data/mnist
addpath svm

% loadMNIST*

images = loadMNISTImages('../data/mnist/train-images-idx3-ubyte')';
labels = loadMNISTLabels('../data/mnist/train-labels-idx1-ubyte')';

% map labels {0, 1, ..., 9} -> {1, 2, ..., 10} 
labels = labels + 1;

% permute input randomly
perm = randperm(size(images, 1));
images = images(perm,:); 
labels = labels(perm);

clear perm;

OPTIMIZE = false;

%% optimize hyperparameters

if OPTIMIZE

N_train = 60000;

Xtrain = images(1:N_train,:);
Ytrain = labels(1:N_train);

svm = get_model("SH", 10);
svm.hyperparams.C = 28;
svm.kernel = "rbf";
svm.hyperparams.gamma = 0.65;

coarse_range = 10.^(-2:2);
fine_range = 2.^(-1:.5:1);
finer_range = .8 : .1 : 1.2;

% good initial search:
% C in logspace(-2, 10, 13);
% gamma in logspace(-9, 3, 13);

param_range = struct();
% param_range.C = svm.hyperparams.C * finer_range;
% param_range.gamma = svm.hyperparams.gamma .* finer_range;

% param_range.d = [1 3 5 7];

tic;
[result, acc_matrix, train_acc_matrix] = optimize_hyperparameters(svm, Xtrain, Ytrain, param_range, 2, 'parallel', 0, 'verbose', 2);
toc

else
    
%% do not optimize hyperparameters -- train and test with one value of them

svm = get_model("AVA", 10);
svm.kernel = "rbf";
svm.hyperparams.C = 90;
svm.hyperparams.gamma = 0.033;

N_train = 60000;

Xtrain = images(1:N_train,:);
Ytrain = labels(1:N_train);


tic
svm = train_model(svm, Xtrain, Ytrain, 'verbose', 2, 'batch_size', 250, 'max_iters', 1000, 'tol', 1e-2, 'batch_method', 'probabilistic');
toc

images_test =  loadMNISTImages('../data/mnist/t10k-images-idx3-ubyte')';
labels_test = loadMNISTLabels('../data/mnist/t10k-labels-idx1-ubyte')';

% map labels {0, 1, ..., 9} -> {1, 2, ..., 10} 
labels_test = labels_test + 1;

test_acc = test_model(svm, images_test, labels_test)

% do jittering (create artificial datapoints)
% a bit of a help. training on half the data with SH SVM went from 93.4%
% accuracy to 94.1%.
JITTER = false;
if JITTER
    [Xtrain,Ytrain] = jitter_mnist(svm.trained.unnormalize(svm.trained.svs), svm.trained.sv_labels);

    % re-train on jittered dataset
    tic
    svmJ = train_model(svm, Xtrain, Ytrain, 'verbose', 2, 'batch_size', 500, 'max_iters', 1000, 'tol', 1e-3);
    toc

    test_acc = test_model(svmJ, images_test, labels_test)
end


end
