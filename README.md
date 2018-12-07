# multiclass-svm

This is a MATLAB implementation of several types of SVM classifiers. 
In addition to the binary SVM, we include six different types of multiclass SVMs.
These are: one-vs-all and all-vs-all based on the binary SVM, the "LLW" classifier presented in [1], the "CS" classifier from [2], 
and the Simplex Halfspace and Simplex Cone SVMs described in [3].

In addition, we provide an extremely efficient quadratic program solver (`solve_qp.m`) that is able to solve optimization problems with a quadratic objective,
linear inequality and equality constraints, and upper and lower bound constraints.
All SVM implementations are based on this solver.

An example of training and testing an SVM on a dataset:

    % Model type is one of "OVA", "AVA", "LLW", "CS", "SC", "SH"
    svm = get_model("SH", K);  % K = number of classes
    svm.hyperparams.C = 5;
    svm.kernel = "rbf";
    svm.hyperparams.gamma = 0.1;

    svm = train_model(svm, Xtrain, Ytrain);

    acc = test_model(svm, Xtest, Ytest);
    
Also included are functions for performing crossvalidation and hyperparameter optimization.

See the script file `SCRIPT_mnist.m` for an example of training hyperparameters and then training a full model and 
testing its accuracy on test data using the MNIST handwriting recognition dataset.


Training the LLW-SVM requires the [CVX Optimization Library](http://cvxr.com/cvx/) in certain cases; all other models are self-contained.


[1] Y. Lee, Y. Lin, and G. Wahba. Multicategory support vector machines. Journal of the American
Statistical Association, 99:465, 67–81, 2004.

[2] K. Crammer and Y. Singer. On the algorithmic implementation of multiclass kernel-based vector
machines. Journal of Machine Learning Research, 2:265–292, 2001.

[3] Y. Mroueh, T. Poggio, L. Rosasco, and J. E. Slotine. Multiclass learning with simplex coding. Advances
in Neural Information Processing Systems, 4, 09 2012.
