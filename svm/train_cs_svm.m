function SVM = train_cs_svm(X, Y, K, kernel_type, hyperparams, varargin)
% Trains Crammer-Singer multi-class SVM

[N, d] = size(X);

[SVM, Xn] = svm_preprocess(X);

kernel = kernels.get(kernel_type);

Kx = @(X1,X2) kernel(X1, X2, hyperparams);

% oneY(:,i) \in R^K = vector of all zeros except in yi's class index
% oneY = zeros(K, N);
% for i = 1:N
%     oneY(Y(i), i) = 1;
% end

% regularizer
% beta = 1; 

kron_obj = @(X1,X2) kron(eye(K), Kx(X1,X2)+1);
yv = @(X,Y)oneY_vec(X, Y, K);

eq_constraint_mat = kron(ones(1,K), eye(N));
% end

% keyboard

% kron_obj = P*kron_obj*P';
% oneY_vec = P*oneY_vec;
% eq_constraint_mat = eq_constraint_mat*P';

Q = @(X1,X2,Y1,Y2)kron_obj(X1,X2);
c = @(X,Y) -hyperparams.C * yv(X,Y);
% A = eye(N*K);
% b = yv(Xn,Y);
A = [];
b = [];
LB = [];
UB = yv(Xn,Y);
D = eq_constraint_mat;

% matlab_qp_tstart = tic;
% [tau,fval,flag] = quadprog(Q(X,X,Y,Y),c(X,Y),A,b,D,zeros(size(D,1), 1), LB, UB);
% fprintf('Quadratic program terminated in %g secs with fval = %g, flag = %d.\n', toc(matlab_qp_tstart), fval, flag); 

%% DIFFERENT LOOP FOR PLOTTING RESULTS
%{
keyboard

qp_soln = zeros(N*K, 1);

output_file = 'iterated_cs.gif';
% output_file = 

for i = 1:5
    [qp_soln, fval] = solve_qp(Q,c,A,b,D,N,K, 'initial_guess', qp_soln, varargin{:});
    tau = reshape(qp_soln, N, K);

    SVM.type = "CS";
    SVM.K = K;

    SVM.sv_indices = find(sum(abs(tau), 2) > 1e-5);
    SVM.tau = tau(SVM.sv_indices, :);
    SVM.svs = Xn(SVM.sv_indices, :);

    SVM.kernel = kernel;
    SVM.hyperparams = hyperparams;

    SVM.predict = @(X) predict_cs_svm(X, SVM);

    h = plot_multiclass_model(SVM, X, Y, 'legend', 0);
    
    frame = getframe(h);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 512);
    
%     if i == 1
%         imwrite(uint8(imind), cm, output_file, 'gif', 'Loopcount', 0);
%     else
%         imwrite(uint8(imind), cm, output_file, 'gif', 'WriteMode', 'append');
%     end

    fname = sprintf('%02d-cs.png', i);
    imwrite(im, fname, 'png');
    
    close;
end

% Write the final result...

[qp_soln, fval] = solve_qp(Q,c,A,b,D,N,K, 'max_iters', 200, 'batch_size', 100);
tau = reshape(qp_soln, N, K);

SVM.type = "CS";
SVM.K = K;

SVM.sv_indices = find(sum(abs(tau), 2) > 1e-5);
SVM.tau = tau(SVM.sv_indices, :);
SVM.svs = Xn(SVM.sv_indices, :);

SVM.kernel = kernel;
SVM.hyperparams = hyperparams;

SVM.predict = @(X) predict_cs_svm(X, SVM);

h = plot_multiclass_model(SVM, X, Y, 'legend', 0);

frame = getframe(h);
im = frame2im(frame);
[imind, cm] = rgb2ind(im, 512);

% imwrite(uint8(imind), cm, output_file, 'gif', 'WriteMode', 'append');

fname = sprintf('%02d-cs.png', i+1);
imwrite(im, fname, 'png');

%}

%% END PLOT LOOP
% 
% [tau, fval] = quadprog(Q(X,X,Y,Y), c(X,Y), A,b,D,zeros(size(D,1),1));
% fprintf('Quadratic program terminated with fval = %g.\n', fval); 

[tau, fval, SVM.train_info] = solve_qp(Q,c,A,b,D,LB,UB,Xn,Y,K,varargin{:});

% cvx_begin
%     variable tau(N*K)
%     
%     minimize(0.5*quad_form(tau, Q) + c'*tau)
%     subject to
%         A*tau <= b;
%         D*tau == 0;
% cvx_end

% cvx_begin
%     variable tau(N*K)
%     
%     minimize( 0.5 * quad_form(tau, kron_obj) - hyperparams.C*oneY_vec'*tau )
%     subject to
%         tau <= oneY_vec;
%         eq_constraint_mat * tau == 0;
% cvx_end
        
% reshape back to N x K form
tau = reshape(tau, N, K);

% cvx_begin
%     variable tau(N,K)
%     
%     obj = 0;
%     for i = 1:K
%         obj = obj - 0.5 * quad_form(tau(:,i), Kx + 1);
%     end
%     
%     maximize( obj + hyperparams.C * trace(tau*oneY) )
%     
%     subject to
%         
%         tau <= oneY';
%         tau * ones(K,1) == 0;
%         
% %         for i=1:N
% %             tau(i,:)' <= oneY(:,i);
% %             tau(i,:) * ones(K,1) == 0;
% %         end
% cvx_end

SVM.type = "CS";
SVM.K = K;

SVM.sv_indices = find(sum(abs(tau), 2) > 1e-5);
SVM.tau = tau(SVM.sv_indices, :);
SVM.svs = Xn(SVM.sv_indices, :);

SVM.kernel = kernel;
SVM.hyperparams = hyperparams;

SVM.predict = @(X) predict_cs_svm(X, SVM);

end

function yv = oneY_vec(X,Y,K)

% oneY(:,i) \in R^K = vector of all zeros except in yi's class index
oneY = zeros(K, size(X,1));
for i = 1:size(X,1)
    oneY(Y(i), i) = 1;
end

yv = reshape(oneY', [], 1);

end
