function model = get_model(type, n_classes)

if strcmpi(type, "SVM_AVA") || strcmpi(type, "AVA")
    model = init_regularized_svm;
    model.type = "SVM_AVA";
elseif strcmpi(type, "SVM_OVA") || strcmpi(type, "OVA")
    model = init_regularized_svm;
    model.type = "SVM_OVA";
elseif strcmpi(type, "SVM_LLW") || strcmpi(type, "LLW")
    model = init_regularized_svm;
    model.type = "SVM_LLW";
elseif strcmpi(type, "SVM_CS") || strcmpi(type, "CS")
    model = init_regularized_svm;
    model.type = "SVM_CS";
elseif strcmpi(type, "SVM_SC") || strcmpi(type, "SC")
    model = init_regularized_svm;
    model.type = "SVM_SC";
elseif strcmpi(type, "SVM_SH") || strcmpi(type, "SH")
    model = init_regularized_svm;
    model.type = "SVM_SH";
end

model.K = n_classes;

end

function model = init_regularized_svm
model.hyperparams.C = 1.0;
model.kernel = "rbf";
model.hyperparams.gamma = 1.0;
end