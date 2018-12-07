function accuracy = test_model(model, Xtest, Ytest, varargin)
% Tests a pre-trained model on a set of test data Xtest with true labels
% Ytest.
%
% Returns the accuracy in [0,1]

preds = model.trained.predict(Xtest);

wrong = preds(:) ~= Ytest(:);

accuracy = 1 - sum(wrong) / numel(wrong);