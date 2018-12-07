function [soln, fval, train_info] = solve_qp(Q, c, A, b, D, LB, UB, X, Y, K, varargin)
% Solves:
% cvx_begin
%     variable a(K*N)
%     minimize(0.5*quad_form(a,Q) + c'*a)
%     subject to
%         A*a <= b
%         LB <= a <= UB
%         D*a == 0
% cvx_end

N = size(X,1);

p = inputParser;
p.KeepUnmatched = true;
p.addParameter('batch_size', 100);
p.addParameter('batch_method', 'relaxation2');
p.addParameter('min_iters', 0);
p.addParameter('max_iters', 250);
p.addParameter('initial_guess', []);
p.addParameter('verbose', 1);
p.addParameter('tol', 1e-3);

parse(p, varargin{:});

verbose = p.Results.verbose;

% try batches of working sets
batch_size = min(N, p.Results.batch_size);

% NOTES:
% QP dual is:
% m = size(A,1); p = size(D,1);
% cvx_begin
%     variable lam(m)
%     variable nu(p)
%     maximize (-0.5*matrix_frac(A'*lam + D'*nu + c, Q) - b'*lam )
%     subject to
%         lam >= 0;
% cvx_end
%
% or equivalently (and more efficiently...):
% 
% cvx_begin
%     variable t(K*N)
%     variable lam(m)
%     variable nu(p)
%     maximize( -0.5*quad_form(t,Q) - b'*lam )
%     subject to
%         lam >= 0;
%         Q*t == A'*lam + D'*nu + c;
% cvx_end
%
%
% KKT conditions for optimality:
% (1) primal feasible
% (2) Q*alpha + c + D'*nu + A'*lam == 0
% (3) lam_i == 0 for i not in A(alpha) (active set)
% (4) dual feasible (lam >= 0)
%
%


% Permute the matrices so batches are easier / more meaningful
% P = zeros(N*K);
% for i=1:N
%     for j=1:K
%         P((i-1)*K + j, i + N*(j-1)) = 1;
%     end
% end
% 
% Q = P*Q*P';
% c = P*c;
% A = A*P';
% D = D*P';

% initialize
if numel(p.Results.initial_guess) == 0
    soln = zeros(N*K,1);
else
    soln = p.Results.initial_guess;
end

if verbose >= 2
    fprintf("Beginning QP solver iterations with %d variables and %d constraints.\n", ...
        N*K, size(A,1)+size(D,1)+size(LB,1)+size(UB,1));
end

% we never want to have to construct the full Q matrix and hence can never
% really compute the full objective function...
% only track relative changes, assume obj(t0) = 0? accurate unless an
% initial guess was provided
objs = [0];
dobjs = [];

train_info = struct();
train_info.iter_times = [0];

% keyboard

% gradient = Q*alpha + c, so at initialization with alpha = 0, ...
gradients = c(X,Y);

% keyboard

% batch_phase = 1;

stall_count = 0;
max_stall = 2;
stall_thresh = p.Results.tol;
% stall_thresh = 1e-3;
% stall_thresh = 1e-2;
qp_tstart = tic;
% keyboard
for i=1:p.Results.max_iters
    iter_tstart = tic;

    if strcmpi(p.Results.batch_method, 'random')
        batch = random_batch(batch_size, Y);
    else
        batch = select_optimal_batch(batch_size, soln, gradients, LB, UB, D, N, K, Y, varargin{:});
    end

    [soln, dobjs(i), dgrad] = solve_batch_problem(batch, soln, Q, c, A, b, D, LB, UB, X, Y, K, varargin{:});
    
    gradients = gradients + dgrad;
    
%     fprintf('new obj = %g\n', objective(soln, Q(X,X,Y,Y), c(X,Y)));

%     objs(i+1) = objective(soln, Q, c);
    
%     dobjs(i+1) = abs(objs(i+1) - objs(i)) / abs(objs(i+1));

    objs(i+1) = objs(i) + dobjs(i);
    
    if verbose >= 2
        fprintf('Iteration %d of QP solver, fval = %g (decrease of %.03f%%).\n', i, objs(i+1), 100*dobjs(i)/objs(i+1));
    end
    
    train_info.iter_times(i+1) = toc(iter_tstart);
    
%     dobjs(i)
    
    if (dobjs(i)/objs(i+1)) < stall_thresh
        stall_count = stall_count + 1;
    else
        stall_count = 0;
    end
    
    if stall_count > max_stall && i >= p.Results.min_iters
        break
    end
end
n_inner_iters = i;
% toc

train_info.objs = objs;
% 
% tic; 
% [soln, fval_min] = quadprog(Q,c,A,b,D,zeros(size(D,1), 1));
% toc

% max_n_epochs = 50;
% soln = zeros(N*K,1);
% objs_epoch = [0];
% dobjs_epoch = [];
% n_inner_iters = 0;
% qp_tstart = tic;
% for epoch = 1:max_n_epochs
%     epoch_tstart = tic;
%     batches = get_epoch_batches(batch_size, N);
%     
%     dobj_inner = 0;
%     
%     for i = 1:numel(batches)
%         [soln, dobjs_epoch(end+1)] = solve_batch_problem(batches{i}, soln, Q, c, A, b, D, LB, UB, X, Y, K, varargin{:});
% %         objs_epoch(end+1) = objective(soln, Q, c);
% %         dobjs_epoch(end+1) = abs(objs_epoch(end) - objs_epoch(end-1)) / abs(objs_epoch(end));
%         dobj_inner = dobj_inner + dobjs_epoch(end);
%         n_inner_iters = n_inner_iters + 1;
%     
%         if verbose >= 3
%             fprintf('Epoch %d, batch %d/%d (iter %d); fval = %g.\n', epoch, i, numel(batches), n_inner_iters, objs(epoch) + dobj_inner);
%         end
%         
%         if n_inner_iters > p.Results.max_iters
%             break
%         end
%     end
%     
%     objs(epoch + 1) = objs(epoch) + dobj_inner;
%     rel_dobj = dobj_inner / objs(epoch + 1);
%     
%     if verbose >= 2
%         fprintf('Epoch %d ended (in %g seconds), obj = %g (decreased by %g%%).\n', epoch, toc(epoch_tstart), objs(epoch+1), 100*rel_dobj);
%     end
%     
%     if rel_dobj < stall_thresh
%         break
%     end
%         
%     if n_inner_iters > p.Results.max_iters
%         break
%     end
% end
% toc

% 
% plot(objs);
% hold on;
% plot(objs_epoch);
% plot(1:numel(objs),  fval_min * ones(1, numel(objs)));
% legend('random', 'epochs', 'exact fval');
    
% fval = objective(soln, Q, c);
fval = objs(end);

% plot(objs_lp); hold on; 
% plot(objs_rnd);

% full batch solution...
% tic; [truth, fval_min] = quadprog(Q,c,A,b,D,zeros(size(D,1), 1)); toc
% fval_min

% un-permute
% soln = P' * soln;

if verbose
    fprintf('Quadratic program terminated in %g seconds after %d iterations with fval = %g.\n', toc(qp_tstart), n_inner_iters, fval); 
end

end

function indices = get_epoch_batches(batch_size, in_size)
if batch_size > in_size
    error('impossible set requested');
end

idxs = randperm(in_size);
for i = 1 : ceil(in_size / batch_size)
    start_idx = 1 + batch_size*(i-1);
    end_idx = min(numel(idxs), start_idx + batch_size - 1);
    indices{i} = idxs(start_idx:end_idx);
end

% for i = 1:ceil(in_size / batch_size)
%     idxs = randperm(in_size);
%     indices{i} = idxs(1:batch_size);
% end

end


function f = objective(soln, Q, c)
f = 0.5 * soln'*Q*soln + c'*soln;
end

