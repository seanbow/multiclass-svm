function SVM = train_binary_svm(X,Y,kernel_type,hyperparams,varargin)
% X is a matrix of training data, m by N
%   -- (m examples of dimension N, one example per row)
%
% Y is an m-dimensional vector of labels in {-1, 1}

inputExist = find(strcmpi(varargin, 'normalize'));
if inputExist
    do_normalize = varargin{inputExist + 1};
else
    do_normalize = true;
end

[m,N] = size(X);

kernel_fun = kernels.get(kernel_type);

% Normalize data, maybe.
[SVM, Xn] = svm_preprocess(X, 'normalize', do_normalize);

% C = 1; % regularizer

%% PRIMAL FORM

% hinge = @(x) max(0, 1 - x);
% 
% cvx_begin
%     variable w(N)
%     variable b
%     
%     minimize( 0.5 * w' * w + C * sum(hinge(Y .* (w'*X' + b) )) )
%     
% cvx_end
% 
% % Find support vectors
% svs = find(Y .* (w'*X' + b) <= 1);
% 
% SVM.w = w;
% SVM.b = b;
% SVM.svs = svs;


%% DUAL FORM
% % 
Kx = @(X1, X2) kernel_fun(X1, X2, hyperparams);

% keyboard

% this large diagonal matrix multiplication is SLOW!
% Q = @(X1, X2, Y1, Y2) diag(Y1)*Kx(X1, X2)*diag(Y2);

% instead...
Q = @(X1, X2, Y1, Y2) Y1' .* Kx(X1, X2) .* Y2;

c = @(X, Y) -ones(size(X,1),1);
% A = [-eye(m) ; eye(m)];
% b = [zeros(m,1) ; hyperparams.C * ones(m,1)];
A = []; b = [];
D = Y;
LB = zeros(m,1);
UB = hyperparams.C * ones(m,1);

% keyboard

[a, fval, SVM.train_info] = solve_qp(Q,c,A,b,D,LB,UB,Xn,Y,1,varargin{:});


% [a,fval,flag] = quadprog(Q,c,A,b,D,zeros(size(D,1), 1), LB, UB);
% fprintf('Quadratic program terminated with fval = %g, flag = %d.\n', fval, flag); 
    
% cvx_begin
%     variable a(m)
%     
%     minimize(0.5*quad_form(a,Q) + c'*a)
%     subject to
%         A*a <= b;
%         D*a == 0;
% cvx_end

% cvx_begin
%     variable a(m)
%     
%     minimize( 0.5 * quad_form(a, diag(Y)*Kx*diag(Y)) - ones(1,m)*a )
%     subject to
%     
%         Y * a == 0;
%         a >= 0;
%         a <= hyperparams.C;
%          
%         
% cvx_end

a = a';

% mid_a = find(a > 1e-8 & a < hyperparams.C - 1e-8);
% b_est = 0;
% for i = mid_a
%     b_est = b_est + Y(i) - a(i)*Y(i)*kernel_fun(X(i,:), X(i,:), hyperparams);
% end
% b_est = b_est / numel(mid_a);

nz_a = find(a > 1e-8);
b_est = 0;
for i = nz_a
    b_est = b_est + Y(i) - a(i)*Y(i)*kernel_fun(X(i,:), X(i,:), hyperparams);
end
b_est = b_est / numel(nz_a);

% keyboard

SVM.b = b_est;

%% SMO!
% [a, b] = smo_train_binary(Xn, Y, kernel_type, hyperparams);
% a = a';
% SVM.b = b;

%%

% SVM.w = sum(a.*Y.*Xn',2);

% mid_a = find(a > 1e-5 & a < C - 1e-5);
% SVM.b = mean(Y(mid_a)' - Xn(mid_a, :)*SVM.w);

SVM.sv_indices = find(a > 1e-5);

SVM.svs = Xn(SVM.sv_indices, :);
SVM.sv_weights = a(SVM.sv_indices);
SVM.sv_classes = Y(SVM.sv_indices);

SVM.kernel = kernel_fun;
SVM.hyperparams = hyperparams;

SVM.predict = @(x) sum( SVM.sv_weights .* SVM.sv_classes .* SVM.kernel(SVM.normalize(x),SVM.svs,SVM.hyperparams), 2 ) + SVM.b;

