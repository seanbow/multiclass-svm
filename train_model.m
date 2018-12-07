function model = train_model(model, X, Y, varargin)

K = model.K;

if strcmpi(model.type, 'svm_ova')
    train_fn = @train_onevsall_svm;
elseif strcmpi(model.type, 'svm_ava')
    train_fn = @train_allvsall_svm;
elseif strcmpi(model.type, 'svm_llw')
    train_fn = @train_llw_svm;
elseif strcmpi(model.type, 'svm_cs')
    train_fn = @train_cs_svm;
elseif strcmpi(model.type, 'svm_sc')
    train_fn = @train_sc_svm;
elseif strcmpi(model.type, 'svm_sh')
    train_fn = @train_sh_svm;
end

model.trained = train_fn(X, Y, K, model.kernel, model.hyperparams, varargin{:});

end