
function batch = select_optimal_batch(batch_size, soln, gradients, LB, UB, D, N, K, Y, varargin)
% g = vector of partial derivatives of the cost function w.r.t. a
% g = Q*soln + c;

p = inputParser;
p.addParameter('batch_method', 'probabilistic');
p.KeepUnmatched = true;

p.parse(varargin{:});

batch_method = p.Results.batch_method;

% HOW do we solve this more efficiently???

% linear program variables = [d ; eta ; xi]
% min f'*x s.t. l<=x<=u, Aeq*x==beq
% One_nk = ones(N*K, 1);
% Zero_nk = zeros(N*K);
% Ink = eye(N*K);
% Alin = [A              zeros(size(A));
%         Ink            Zero_nk;
%         -Ink           Zero_nk;
%         Zero_nk        -Ink;
%         zeros(1, N*K)  One_nk';
%         Ink            -Ink;
%         -Ink           -Ink];
%     
% blin = [b - A*soln;
%         One_nk    ;
%         One_nk;
%         zeros(N*K, 1);
%         batch_size;
%         zeros(N*K, 1);
%         zeros(N*K, 1)];

% sets of active ineq constraints
% ineq_active = find(abs(b-A*soln) < 1e-6);
% keyboard
if numel(LB) == 0
    L_active = [];
else
    L_active = find(abs(soln - LB) < 1e-8);
end

if numel(UB) == 0
    U_active = [];
else
    U_active = find(abs(soln - UB) < 1e-8);
end

lin_lb_d = -1 * ones(N*K, 1);
lin_lb_d(L_active) = 0;

lin_ub_d = ones(N*K, 1);
lin_ub_d(U_active) = 0;

lin_lb_eta = zeros(N, 1);
lin_ub_eta = Inf * ones(N,1);

lin_lb_xi = -Inf * ones(N,1);
lin_ub_xi = Inf * ones(N,1);

lin_lb = [lin_lb_d ; lin_lb_eta ; lin_lb_xi];
lin_ub = [lin_ub_d ; lin_ub_eta ; lin_ub_xi];

% select rows of A corresponding only to active constraints
% Aac = A(ineq_active, :);
% bac = b(ineq_active, :);

if strcmpi(batch_method, "optimal")

    Delta = kron(eye(N), ones(K,1));

    % varaibles: d, eta, xi.
    % tota

    Alin = [eye(N*K)       -Delta         zeros(N*K,N) ;
            -eye(N*K)      -Delta         zeros(N*K,N) ;
            zeros(N,N*K)    eye(N)        -eye(N)      ;
            zeros(N,N*K)    -eye(N)       -eye(N)      ;
            zeros(1,N*K)    zeros(1,N)    ones(1, N)   ];

    blin = [ zeros(N*K, 1);
             zeros(N*K, 1);
             zeros(N, 1)  ;
             zeros(N, 1)  ;
             batch_size   ];


    if numel(D) > 0
        Aeq_lin = [D    zeros(size(D,1), N)    zeros(size(D,1), N)];
        beq_lin = zeros(size(D,1), 1);
        problem.Aeq = Aeq_lin;
        problem.beq = beq_lin;
    end

    problem.f = [gradients ; zeros(N, 1) ; zeros(N,1)];
    problem.lb = lin_lb;
    problem.ub = lin_ub;
    problem.Aineq = Alin;
    problem.bineq = blin;
    problem.solver = 'linprog';
    problem.options = optimoptions('linprog');
    % problem.options.Algorithm = 'dual-simplex';
    problem.options.Algorithm = 'interior-point';
    problem.options.Display = 'off';

    [xopt, fval, exitflag, output] = linprog(problem);

    eta = xopt(N*K + 1 : N*K + N);
    batch = find(abs(eta) > 1e-4)';

    % how to fix this?? what if the batch is all from one class
    if numel(unique(Y(batch))) < K
        % pick some others...
        % this is a shitty method
        diff_idxs = find(Y ~= Y(batch(1)));
        diff_idxs = diff_idxs(randperm(numel(diff_idxs)));

        batch = [batch diff_idxs(1:min(batch_size, numel(diff_idxs)))];
        batch = batch(randperm(numel(batch)));
        batch = batch(1:batch_size);
    end

elseif strcmpi(batch_method, "relaxation_1") || strcmpi(batch_method, "relaxation1")

    % test heuristic
    % instead of d <= Delta * eta and |eta|_1 <= q, just try 
    % |d|_1 <= K * q;
    Aln = [-eye(K*N)       -eye(K*N)     ;
            eye(K*N)       -eye(K*N)     ;
           zeros(1, K*N)   ones(1, K*N) ];

    bln = [zeros(K*N, 1);
           zeros(K*N, 1);
           K * batch_size];

    problem = struct();
    problem.f = [gradients ; zeros(N*K, 1)];
    problem.Aineq = Aln;
    problem.bineq = bln;
    problem.lb = [lin_lb_d ; -Inf * ones(N*K, 1)];
    problem.ub = [lin_ub_d ; Inf * ones(N*K, 1)];
    problem.solver = 'linprog';
    problem.options = optimoptions('linprog');
    % problem.options.Algorithm = 'dual-simplex';
    problem.options.Algorithm = 'interior-point';
    problem.options.Display = 'off';

    [xopt, fval, exitflag, output] = linprog(problem);

    d = xopt(1:N*K);

    % reconstruct fake "eta"
    d_nonzero = find(abs(d) > 1e-4);
    eta_nz = unique(mod(d_nonzero - 1, N) + 1)';
    batch = eta_nz;
    
elseif strcmp(batch_method, "relaxation_2") || strcmpi(batch_method, "relaxation2")

    % try again: use -abs(gradients) and assume d >= 0
    % here the "true" direction at coordinate i might have d_i < 0 even if
    % the solution here is d_i > 0
    %  --> need to redo the active lower bound constraint. if the gradient
    %  is > 0 for i in LB_active, set the upper bound to 0.
%     lin_ub_d(gradients > 0 & abs(soln-LB) < 1e-8) = 0;
   
    lin_ub_d = ones(N*K, 1);
    if numel(LB) > 0
        lin_ub_d(gradients > 0 & abs(soln-LB) < 1e-6) = 0;
    end
    if numel(UB) > 0
        lin_ub_d(gradients < 0 & abs(soln-UB) < 1e-6) = 0;
    end
    
    problem = struct();
    problem.f = -abs(gradients);
    problem.Aineq = ones(1, K*N);
    problem.bineq = K * batch_size;
    problem.lb = zeros(K*N, 1);
    problem.ub = lin_ub_d;
    problem.solver = 'linprog';
    problem.options = optimoptions('linprog');
    % problem.options.Algorithm = 'dual-simplex';
    problem.options.Algorithm = 'interior-point';
    problem.options.Display = 'off';

    % tic
    [d, fval, exitflag, output] = linprog(problem);
    % toc

    % reconstruct fake "eta"
    d_nonzero = find(abs(d) > 1e-4);
    eta_nz = unique(mod(d_nonzero - 1, N) + 1)';
    batch = eta_nz;

    % how to fix this?? what if the batch is all from one class
    if numel(unique(Y(batch))) < K
        % pick some others...
        % this is a shitty method
        diff_idxs = find(Y ~= Y(batch(1)));
        diff_idxs = diff_idxs(randperm(numel(diff_idxs)));

        batch = [batch diff_idxs(1:min(batch_size, numel(diff_idxs)))];
        batch = batch(randperm(numel(batch)));
        batch = batch(1:batch_size);
    end

elseif strcmpi(batch_method, "probabilistic")
    valid_set = ones(N*K, 1);
    if numel(LB > 0), valid_set(gradients > 0 & abs(soln-LB) < 1e-8) = 0; end
    if numel(UB > 0), valid_set(gradients < 0 & abs(soln-UB) < 1e-8) = 0; end
    
    batch = [];
    
    if sum(valid_set) > 0

        probs = gradients;
        probs(valid_set == 0) = 0;

    %     sample_probs = sum(reshape(abs(probs), N, K), 2);
        sample_probs = sqrt(sum(reshape(probs, N, K).^2, 2));
    %     sample_probs = sample_probs / sum(sample_probs);

        actual_batch_size = min([batch_size, sum(sample_probs > 0), N]);

        if sum(sample_probs) == 0
            keyboard
        end

        batch = datasample(1:N, actual_batch_size, 'Replace', false, 'Weights', sample_probs);
    
        % if the batch "failed" in the sense that only one class was sampled,
        % force class-equality by sampling from each class independently...?
    %     keyboard
        if numel(unique(Y(batch))) < K
            size_per_class = ceil(actual_batch_size / K);
            batch_per_class = {};
            for i=1:K
                batch_per_class{i} = datasample(find(Y == i), size_per_class, 'Replace', false, 'Weights', sample_probs(Y == i));
            end
            batch = cat(2, batch_per_class{:});
            batch = batch(1:actual_batch_size);
        end
    end
    
    % fill the rest randomly...?
    if numel(batch) < min(batch_size, N)
        candidates = setdiff(1:N, batch);
        remainder = random_batch(min(batch_size, N) - numel(batch), Y(candidates));
        batch = [batch candidates(remainder)];
    end
    
else
    error("unknown batch selection method!!");
end

if numel(batch) > batch_size
    % can happen sometimes because we're approximating the 0-norm
    perm = randperm(numel(batch));
    batch = batch(perm);
    batch = batch(1:batch_size);
end

% add some random ones
% not_in_batch = setdiff(1:N, batch);
% not_in_batch = not_in_batch(randperm(numel(not_in_batch)));
% batch = [not_in_batch(1:0.1*batch_size) batch];
% batch = batch(1:min(batch_size, numel(batch)));

% alpha_batch = find(abs(d) > 1e-4)';
% batch = mod(alpha_batch-1,N)+1;
% batch = unique(batch);
% if numel(batch) > batch_size
%     batch = batch(1:batch_size);
% end

% [gsorted,sort_idx] = sort(g, 'descend');
% sort_idx = unique(mod(sort_idx - 1, N) + 1);
% batch = [sort_idx(1:batch_size); sort_idx(end-batch_size+1 : end)]';

end