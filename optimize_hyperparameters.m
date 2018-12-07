function [result, acc_matrix, train_acc_matrix] = optimize_hyperparameters(model, X, Y, parameter_ranges, numOfFolds, varargin)
% Optimizes a set of hyperparameters for a given model 'model'
%
% Parameter_ranges must be a struct of parameter names and the set over
% which to optimize. For example:
%  parameter_ranges.C: [0.1 1 10 100];
%  parameter_ranges.gamma: [.25 .5 1 2 4 8]

p = inputParser;
p.KeepUnmatched = true;
p.addParameter('parallel', 0);
p.addParameter('verbose', 1);

parse(p, varargin{:});

if p.Results.parallel
    n_threads = Inf;
else
    n_threads = 0;
end

%% default to three folds
if ~exist('numOfFolds','var')
  numOfFolds=3;
end

param_names = fieldnames(parameter_ranges);

% Perform a grid search
% TODO how to do this without cases

input.model = model;
input.X = X;
input.Y = Y;

if numel(param_names) == 1
    
%     grid = ndgrid(parameter_ranges.(param_names{1}));
%     results = arrayfun(@(p1)check_params(input, param_names, p1), grid);
    results = zeros(numel(parameter_ranges.(param_names{1})), 1);
    train_accs = zeros(numel(parameter_ranges.(param_names{1})), 1);
    parfor (i = 1:numel(parameter_ranges.(param_names{1})), n_threads)
        [results(i), train_accs(i)] = check_params(numOfFolds, input, param_names, parameter_ranges.(param_names{1})(i), varargin{:});
    end
    
elseif numel(param_names) == 2
    
    train_accs = zeros(numel(parameter_ranges.(param_names{1})), numel(parameter_ranges.(param_names{2})));
    results = zeros(size(train_accs));
    
%     parfor (ix = 1:numel(results), n_threads)
    for ix = 1:numel(results)
        [a,b] = ind2sub(size(train_accs), ix);
        results(ix) = check_params(numOfFolds, input, param_names, ...
                                    parameter_ranges.(param_names{1})(a), ...
                                    parameter_ranges.(param_names{2})(b), varargin{:});
    end
    
elseif numel(param_names) == 3
    
%     [g1, g2, g3] = ndgrid(parameter_ranges.(param_names{1}), ...
%                           parameter_ranges.(param_names{2}), ...
%                           parameter_ranges.(param_names{3}));
%     results = arrayfun(@(p1,p2,p3)check_params(input, param_names, p1, p2, p3), g1, g2, g3);
    
    results = zeros(numel(parameter_ranges.(param_names{1})), ...
                    numel(parameter_ranges.(param_names{2})), ...
                    numel(parameter_ranges.(param_names{3})));
    train_accs = zeros(numel(parameter_ranges.(param_names{1})), ...
                    numel(parameter_ranges.(param_names{2})), ...
                    numel(parameter_ranges.(param_names{3})));
                
                
    parfor (i = 1:numel(parameter_ranges.(param_names{1})), n_threads)
        results_i = zeros(numel(parameter_ranges.(param_names{2})), ...
                          numel(parameter_ranges.(param_names{3})));
        train_accs_i = zeros(numel(parameter_ranges.(param_names{2})), ...
                          numel(parameter_ranges.(param_names{3})));                      
                      
        for j = 1:numel(parameter_ranges.(param_names{2}))
            for k = 1:numel(parameter_ranges.(param_names{3}))
                
                [results_i(j,k), train_accs_i(j,k)] = check_params(numOfFolds, input, param_names, ...
                    parameter_ranges.(param_names{1})(i), ...
                    parameter_ranges.(param_names{2})(j), ...
                    parameter_ranges.(param_names{3})(k), varargin{:});
            end
        end
        
        results(i,:,:) = results_i;
        train_accs(i,:,:) = train_accs_i;
    end
    
else
    error('too many hyperparameters');
end

[maxval, ind] = max(results(:));

result.accuracy = maxval;

acc_matrix = results;
train_acc_matrix = train_accs;

if numel(param_names) == 1
    result.params.(param_names{1}) = parameter_ranges.(param_names{1})(ind);
elseif numel(param_names) == 2
    [i1, i2] = ind2sub(size(results), ind);
    result.params.(param_names{1}) = parameter_ranges.(param_names{1})(i1);
    result.params.(param_names{2}) = parameter_ranges.(param_names{2})(i2);
elseif numel(param_names) == 3
    [i1, i2, i3] = ind2sub(size(results), ind);
    result.params.(param_names{1}) = parameter_ranges.(param_names{1})(i1);
    result.params.(param_names{2}) = parameter_ranges.(param_names{2})(i2);
    result.params.(param_names{3}) = parameter_ranges.(param_names{3})(i3);
else
    error('too many hyperparameters');
end

end
    

function [result, train_acc] = check_params(numOfFolds, input, names, varargin)

fprintf("Testing parameters ");
for i=1:numel(names)
    fprintf("%s=%f ", names{i}, varargin{i});
end
fprintf("\n");

for i = 1:numel(names)
    input.model.hyperparams.(names{i}) = varargin{i};
end

[result, train_acc] = crossvalidate(input.model, input.X, input.Y, numOfFolds, varargin{numel(names)+1:end});

end