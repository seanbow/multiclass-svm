function SVM = train_sh_svm(X, Y, K, kernel_type, hyperparams, varargin)
% Trains simplex halfspace multi-class SVM

% keyboard

[N, d] = size(X);

[SVM, Xn] = svm_preprocess(X);

kernel = kernels.get(kernel_type);

Kx = @(X1,X2)kernel(X1, X2, hyperparams);

code = simplex_code(K);

% Build matrix G, G(i,j) = code(:,i)' * code(:,j)
G = code'*code;

GY = @(Y1,Y2)G(Y1,Y2);
% 
% cvx_begin
%     variable alp(N);
% %     expression obj;
% %     
% %     obj = 0;
% %     for i=1:N
% %         for j=1:N
% %             obj = obj - 0.5*alp(i)*Kx(i,j)*G(Y(i),Y(j))*alp(j);
% %         end
% %     end
%     
%     minimize( 0.5*quad_form(alp, Kx.*GY) - ones(1, N)*alp )
%     subject to
%         alp >= 0;
%         alp <= 1 / (2 * N * hyperparams.C);
% cvx_end

% keyboard

Q = @(X1,X2,Y1,Y2) Kx(X1,X2).*GY(Y1,Y2);
c = @(X,Y) -ones(size(X,1),1);

% A = [-eye(N);
%      eye(N)];
% b = [zeros(N,1);
%      ones(N,1) / (2*N*hyperparams.C)];
A = [];
b = [];
D = [];

LB = zeros(N,1);
UB = ones(N,1) / (2*N*hyperparams.C);
% UB = ones(N,1) / (2*hyperparams.C);

% alp = quadprog(Q, c, A, b);
[alp, fval, SVM.train_info] = solve_qp(Q,c,A,b,D,LB,UB,Xn,Y,1,varargin{:});

% [alp_true, fval_true] = quadprog(Q(X,X,Y,Y),c(X,Y),A,b,D,zeros(size(D,1), 1));
% fprintf('fval true = %g\n', fval_true);

% Q = kron_obj;
% c = -hyperparams.C * oneY_vec;
% A = eye(N*K);
% b = oneY_vec;
% D = eq_constraint_mat;
% 
% [tau, fval] = solve_qp(Q,c,A,b,D,N,K,varargin{:});
% fprintf('Quadratic program terminated with fval = %g.\n', fval); 

% keyboard

% representer theorem coeffs: a_i = -sum_{y \neq yi} alp(i,y) code(:,y)
% but note alp(i,yi) = 0 for all i so also:
%    a_i = -sum_i alp(i,y) code(:,y)
%
% then f(x) = sum_i a_i k(x, x_i)

% determine a cutoff threshold where we assume alp that are less than it
% are actually zero...
threshold = min(1e-8, UB(1) * 1 / 100);

inds = find(alp > threshold);
SVM.sv_indices = inds;
SVM.svs = Xn(inds, :);
SVM.sv_labels = Y(inds);
SVM.a = alp(inds)' .* code(:, Y(inds));

SVM.code = code;


SVM.type = "SH";
SVM.K = K;

SVM.kernel = kernel;
SVM.hyperparams = hyperparams;

SVM.predict = @(X) predict_sc_svm(X, SVM);


