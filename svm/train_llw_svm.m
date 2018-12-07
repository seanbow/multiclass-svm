function SVM = train_llw_svm(X, Y, K, kernel_type, hyperparams, varargin)

[N,m] = size(X); % note N is number of samples!

METHOD = "DUAL";
% PRIMAL solves the primal optimization
% DUAL solves the dual
%
% Both produce the same *objective* value but may
% produce different classifiers...
%
% Dual is supposed to be faster but it isn't always on small problems

[SVM,Xn] = svm_preprocess(X);

kernel = kernels.get(kernel_type);

Kx = @(X1, X2) kernel(X1, X2, hyperparams);

% Matrix L
% Row i = L(yi) = a vector equal to 0 at index yi and 1 elsewhere
L = ones(size(X,1), K);
for i = 1:size(X, 1)
    L(i, Y(i)) = 0;
end

% Matrix Y in the paper...
% The ith row is 1 at index Y(i), -1/(K-1) elsewhere.
Y_encoded = ones(size(X,1), K) * (-1 / (K-1));
for i = 1:size(X,1)
    Y_encoded(i, Y(i)) = 1;
end

% regularization
% C = 10;
% lambda = 1;

%% PRIMAL
if strcmpi(METHOD, "PRIMAL")
    cvx_begin
        variable xi(N,K)
        variable c(N,K)
        variable b(K)

        obj = 0;
        for j=1:K
            obj = obj + ...
                hyperparams.C * L(:,j)' * xi(:,j) + ...
                0.5 * quad_form(c(:,j), Kx);
        end

        minimize(obj)

        subject to

            for j = 1:K
                b(j)*ones(N,1) + Kx*c(:,j) - Y_encoded(:,j) <= xi(:,j);
                xi(:,j) >= 0;
            end

            sum(b)*ones(N,1) + Kx*sum(c,2) == 0;
    cvx_end

%% DUAL FORM
else

    kron_quad_form = @(X1, X2) kron(eye(K) - ones(K)/K, Kx(X1,X2));
    kron_constraint = kron(eye(K) - ones(K)/K, ones(1,N));

    % Flatten into vectors for optimization purposes
    L_vec = reshape(L, [], 1);
%     Y_vec = reshape(Y_encoded, [], 1);
    
    Q = @(X1, X2, Y1, Y2) kron_quad_form(X1,X2);
    c = @(X,Y) Y_encoding_vector(X,Y,K);
%     A = [-eye(N*K) ; eye(N*K)];
%     b = [zeros(N*K,1) ; L_vec/hyperparams.C];
    A = []; b = [];
    LB = zeros(N*K, 1);
    UB = L_vec/hyperparams.C;
    D = kron_constraint;
    
    [a, fval, SVM.train_info] = solve_qp(Q,c,A,b,D,LB,UB,Xn,Y,K,varargin{:});
    
%     
%     [a, fval,flag] = quadprog(Q(Xn,Xn,Y,Y),c(Xn,Y),A,b,D,zeros(size(D,1), 1));
%     fprintf('Quadratic program terminated with fval = %g, flag = %d.\n', fval, flag); 
    
%     cvx_begin
%         variable a(N*K)
% 
%         minimize(0.5*quad_form(a,Q) + c'*a)
%         subject to
%             A*a <= b;
%             D*a == 0;
%     cvx_end

%     cvx_begin
%         variable a(N*K,1) 
% 
%         minimize( 0.5 * quad_form(a, kron_quad_form) + Y_vec' * a )
%         subject to
% 
% %             a >= 0;
% %             a <= hyperparams.C * L_vec;
%             kron_constraint * a == 0;
%             
%             [-eye(N*K) ; 
%               eye(N*K) ] * a <= [zeros(N*K,1) ; 
%                                  hyperparams.C * L_vec];

%     cvx_end

    % EQUIVALENT DUAL BUT SLOWER :
%     cvx_begin
%         variable a(N, K)
% 
%         obj = 0;
%         for j = 1:K
%             obj = obj + ...
%                 0.5*quad_form(a(:,j)-sum(a,2)/K, Kx) + ...
%                 hyperparams.C*a(:,j)'*Y_encoded(:,j);
%         end
% 
%         minimize(obj)
% 
%         subject to
% 
%             a >= 0
% 
%             for j = 1:K
%                 a(:,j) <= L(:,j);
%                 sum(a(:,j) - sum(a,2)/K) == 0;
%             endtrained
%     cvx_end

    % Find coefficients c
    A_mat = reshape(a, N, []);
    abar = sum(A_mat, 2) / K;
    c = -bsxfun(@minus, A_mat, abar);% / (C);

    % Compute hi vectors
    % h(i,k) = sum_l c(l,k)*K(l,i)
    h = zeros(N, K);

    % find coefficients b
    good_a = A_mat > 1e-5 & A_mat < L - 1e-5;
    good_a = sum(good_a, 2);
    good_a = find(good_a);

    b = zeros(K, 1);
    
%     keyboard

    if numel(good_a) > 0
%         numel(good_a)
        
%         for ix = 1:numel(good_a)
%             i = good_a(ix);
%             Kxi = Kx(Xn, Xn(i,:));
%             for k = 1:K
%     %             h(i,k) = sum(c(:,k) .* Kx(:, i));
%                 h(i,k) = sum(c(:,k) .* Kxi);
%             end
%         end
%         
%         b = mean(Y_encoded(good_a, :) - h(good_a, :), 1)';
        
        %% %% %%
        
        % randomly sample some points
        n_samples = min(50, numel(good_a));
        
        h = zeros(n_samples, K);
        perm = randperm(numel(good_a));
        good_a = good_a(perm);
        for ix = 1:n_samples
            i = good_a(ix);
            Kxi = Kx(Xn, Xn(i,:));
            for k = 1:K
                h(ix,k) = sum(c(:,k) .* Kxi);
            end
        end
        
        b = mean(Y_encoded(good_a(1:n_samples), :) - h, 1)';
        
        %% %% %% 
            
    else
        % Have to solve another optimization...
        % this shouldn't happen often. seems only in sort of degenerate
        % cases with very few data points or if it's way over-regularized.
        clamp = @(x) max(0, x);
        
        % need ALL h here...
        for i = 1:N
            Kxi = Kx(Xn, Xn(i,:));
            for k = 1:K
    %             h(i,k) = sum(c(:,k) .* Kx(:, i));
                h(i,k) = sum(c(:,k) .* Kxi);
            end
        end

        cvx_begin
            variable b(K)

            objective = 0;
            for i = 1:N
                objective = objective + L(i,:) * clamp(h(i,:)' + b - Y_encoded(i,:)');
            end

            minimize( objective )
            subject to

                b' * ones(K,1) == 0;
        cvx_end
    end
end

%} 
%% END DUAL FORM

SVM.type = "LLW";

SVM.K = K;

% keyboard

% "support vector" equivalents in this multiclass setting
SVM.sv_indices = find(sum(abs(c), 2) > 1e-5);
SVM.svs = Xn(SVM.sv_indices, :);
SVM.sv_classes = Y(SVM.sv_indices);

SVM.b = b;
SVM.c = c(SVM.sv_indices, :);

SVM.kernel = kernel;
SVM.hyperparams = hyperparams;

SVM.predict = @(X) predict_llw_svm(X, SVM);

end

function yv = Y_encoding_vector(X, Y, K)

% Matrix Y in the paper...
% The ith row is 1 at index Y(i), -1/(K-1) elsewhere.
Y_encoded = ones(size(X,1), K) * (-1 / (K-1));
for i = 1:size(X,1)
    Y_encoded(i, Y(i)) = 1;
end

yv = reshape(Y_encoded, [], 1);
end

