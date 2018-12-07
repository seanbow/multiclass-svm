function SVM = train_sc_svm(X, Y, K, kernel_type, hyperparams, varargin)
% Trains simplex cone multi-class SVM

% keyboard

[N, d] = size(X);

[SVM, Xn] = svm_preprocess(X);

kernel = kernels.get(kernel_type);

Kx = @(X1,X2) kernel(X1, X2, hyperparams);

code = simplex_code(K);

% Build matrix G, G(i,j) = code(:,i)' * code(:,j)
G = code'*code;

H = @(X1,X2)kron(G,Kx(X1,X2));

% build matrix of Y indicators...
% M(i,j) = 0 if Y(i) == j
% M(i,j) = 1 else
M = ones(N,K);
for i = 1:N
    M(i,Y(i)) = 0;
end

Mvec = M(:);

% cvx_begin
%     variable alp(N*K);
%     
%     maximize( -0.5*quad_form(alp, H) + ones(1, N*K)*alp / (K-1) )
%     subject to
%         alp >= 0;
%         alp <= Mvec / (2 * N * hyperparams.C);
% cvx_end

% c_orig = -ones(N*K, 1) / (K-1);
% I_NK = eye(N*K);
        
Q = @(X1,X2,Y1,Y2)H(X1,X2);
% c = @(X,Y) c_orig(1:(K*size(X,1)));
c = @(X,Y) -ones(K*size(X,1), 1) / (K-1);

% A = [-I_NK;
%       I_NK];
% b = [zeros(N*K,1);
%      Mvec/(2*N*hyperparams.C)];
A = [];
b = [];
D = [];

LB = zeros(N*K, 1);
UB = Mvec / (2*N*hyperparams.C);

[alp, fval, SVM.train_info] = solve_qp(Q,c,A,b,D,LB,UB,Xn,Y,K,varargin{:});

% [alp_true, fval_true] = quadprog(Q(Xn,Xn,Y,Y),c(Xn,Y),A,b,D,zeros(size(D,1), 1));
% fprintf('fval true = %g\n', fval_true);

% keyboard

% cvx_begin
%     variable alp(N,K);
%     expression obj;
%     
%     obj = 0;
%     for i = 1:N
%         for j = 1:N
%             obj = obj - 0.5 * Kx(i,j) * alp(i,:) * G * alp(j,:)';
%         end
%     end
%     
%     for i = 1:N
%         for p = 1:K
%             obj = obj + alp(i,p) / (K-1);
%         end
%     end
%     
%     maximize( obj )
%     subject to
%         alp >= 0;
%         
%         for i = 1:N
%             for p = 1:K
%                 if p == Y(i)
%                     alp(i,p) <= hyperparams.C;
%                 else
%                     alp(i,p) <= 0;
%                 end
%             end
%         end
% cvx_end

% Q = kron_obj;
% c = -hyperparams.C * oneY_vec;
% A = eye(N*K);
% b = oneY_vec;
% D = eq_constraint_mat;
% 
% [tau, fval] = solve_qp(Q,c,A,b,D,N,K,varargin{:});
% fprintf('Quadratic program terminated with fval = %g.\n', fval); 

        
% reshape back to N x K form
alp = reshape(alp, N, K);

% representer theorem coeffs: a_i = -sum_{y \neq yi} alp(i,y) code(:,y)
% but note alp(i,yi) = 0 for all i so also:
%    a_i = -sum_i alp(i,y) code(:,y)
%
% then f(x) = sum_i a_i k(x, x_i)
inds = find(sum(abs(alp), 2) > 1e-5);
SVM.sv_indices = inds;
SVM.svs = Xn(inds, :);
SVM.a = -code * alp(inds,:)';

SVM.code = code;


SVM.type = "SC";
SVM.K = K;

SVM.kernel = kernel;
SVM.hyperparams = hyperparams;

SVM.predict = @(X) predict_sc_svm(X, SVM);

