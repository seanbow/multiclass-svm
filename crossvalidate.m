function [result, train_acc] = crossvalidate(model, X, Y, nfolds, varargin)
% Performs k-fold cross validation of a given model on a dataset X,Y.
%
% Returns the average test accuracy over all folds
%
% If number of folds passed is 1 then trains on whole data and tests on
% whole data
%
if nfolds == 1
    train_acc = 0;
    model = train_model(model, X, Y);
    result = test_model(model, X, Y);
    fprintf("Did not perform cross validation. Just trained and test model with values supplied. Accuracy = %.3g%%.\n", 100*train_acc);

    
else

    inds = crossvalidation_indices(X, nfolds);

    result = 0;
    train_acc = 0;

    for fold = 1:nfolds
        Xtrain = X(inds ~= fold, :);
        Ytrain = Y(inds ~= fold);

        Xtest = X(inds == fold, :);
        Ytest = Y(inds == fold);

        fprintf("Training fold %d... ", fold);

        model = train_model(model, Xtrain, Ytrain, varargin{:});
        train_acc = test_model(model, Xtrain, Ytrain, varargin{:});
        acc = test_model(model, Xtest, Ytest, varargin{:});

    %     fprintf("done. Train accuracy = %.3f%%, Test accuracy = %.3f%%.\n", 100*train_acc, 100*acc);
        fprintf("done. Test accuracy = %.3g%%.\n", 100*acc);

        result = result + acc;
        train_acc = train_acc + train_acc;
    end

    result = result / nfolds;
    train_acc = train_acc / nfolds;

end