function [soln, dobj, dgrad] = solve_batch_problem(batch, soln, Q, c, A, b, D, LB, UB, X, Y, K, varargin)

N = size(X,1);

batch = sort(batch);

% Extract relevant portions of matrices
% keyboard


K_spread = (1:K) - 1;
K_spread = K_spread * N;
K_spread = K_spread' * ones(1, numel(batch));

B_indices = ones(K,1)*batch + K_spread;
B_indices = B_indices(:);
B_indices = sort(B_indices); % necessary so X and alpha follow same indexing... TODO figure this out better
% 
% K_spread = (1:K) - 1;
% K_spread = K_spread' * ones(1, numel(batch));
% 
% indices_begin = K*(batch - 1) + 1;
% 
% B_indices = ones(K,1)*indices_begin + K_spread;
% B_indices = B_indices(:);

N_indices = setdiff(1:(K*N), B_indices);

% find upper-bound indices...
alphab = soln(B_indices);
alphan = soln(N_indices);

if numel(UB) > 0
    % todo pick better threshold
    subbatch = find(abs(alphab - UB(B_indices)) > 1e-10);
    
    % TODO handle this -- optimize only over those in subbatch?
    % for now just break if there are none...
    if numel(subbatch) == 0
        dobj = 0;
        dgrad = zeros(N*K,1);
        return;
    end
end

% keyboard

Xn_indices = setdiff(1:N, batch);

Xb = X(batch, :);
Xn = X(Xn_indices, :);
Yb = Y(batch);
Yn = Y(Xn_indices);

Qbb = Q(Xb, Xb, Yb, Yb);
cb = c(Xb, Yb);

Qbn = Q(Xb, Xn, Yb, Yn);
% An = A(:, N_indices);

% The construction of the matrix An takes *a lot* of time in large problems
%
% hack to get around this: we only need it in the form An*alphan, so
% instead perform the multiplication with the matrix we already have (A) by
% setting entries of alpha(B) = 0 and doing A*alpha
% keyboard
if size(A,1) > 0
    alp_bzero = soln;
    alp_bzero(B_indices) = 0;
    Analphan = A*alp_bzero;
    
    Ab = A(:, B_indices);
    
    problem.Aineq = Ab;
    % problem.bineq = b - An * alphan;
    problem.bineq = b - Analphan;
end

if numel(LB) > 0
    LBb = LB(B_indices);
    problem.lb = LBb;
end

if numel(UB) > 0
    UBb = UB(B_indices);
    problem.ub = UBb;
end
    
if numel(D) > 0
    Db = D(:, B_indices);
    Dn = D(:, N_indices);
    problem.Aeq = Db;
    problem.beq = -Dn*alphan;
end

scale = 1;
problem.H = (Qbb+Qbb')/2;
problem.f = Qbn * alphan + cb;

% scale = mean(Qbb(:)) / 2;
% problem.H = (Qbb+Qbb') / (2*scale);
% problem.f = (Qbn * alphan + cb) / scale;

problem.solver = 'quadprog';

problem.x0 = alphab;

problem.options = optimoptions('quadprog');
problem.options.OptimalityTolerance = 1e-10;
problem.options.Display = 'off';

old_obj = objective(alphab, alphan, Qbb, Qbn, cb);

% keyboard

[batch_soln, fval, flag] = quadprog(problem);

fval = fval * scale;

% fval

% keyboard

% batch_soln = quadprog(Qbb,              ...
%                       Qbn*alphan + cb, ...
%                       Ab,               ...
%                       b - An*alphan,    ...
%                       Db,               ...
%                       -Dn*alphan);

if (flag == 0 || flag == 1) && (fval < 0)

    % Compute gradient update
    dgrad = zeros(N*K, 1);
    dgrad(B_indices) = Qbb * (batch_soln - alphab);
    dgrad(N_indices) = Qbn' * (batch_soln - alphab);
    
    soln(B_indices) = batch_soln;
    dobj = fval - old_obj;
    
%     true_grad = zeros(N*K, 1);
%     true_grad(B_indices) = Qbb*soln(B_indices) + Qbn*soln(N_indices);
%     true_grad(N_indices) = Qbn'*soln(B_indices) + Q(Xn,Xn,Yn,Yn)*soln(N_indices);
%     true_grad = true_grad + c(X,Y);
else
    dobj = 0;
    dgrad = zeros(N*K,1);
end
                  
end

function f = objective(alphab, alphan, Qbb, Qbn, cb)
f = 0.5*alphab'*Qbb*alphab + (Qbn * alphan + cb)'*alphab;
end