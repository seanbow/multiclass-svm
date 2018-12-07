
function st = random_batch(out_size, Y)
% Select a batch of size out_size from a dataset making sure to sample some
% from each class

% K = max(Y);
uniqueY = unique(Y);
K = numel(uniqueY); % in binary SVMs y \in [-1, 1], not [1, 2]...

if out_size > numel(Y)
    error('impossible set requested');
end

per_class = ceil(out_size / K);

% keyboard

% collect indices from each class
class_inds = {};
class_inds_permuted = {};
for i = 1:K
    class_inds{i} = find(Y == uniqueY(i));
    permuted = randperm(numel(class_inds{i}));
    class_inds_selected{i} = permuted(1:min(per_class, numel(permuted)));
end

idxs = [];
for i=1:K
    idxs = [idxs class_inds{i}(class_inds_selected{i})];
end

random_order = randperm(numel(idxs));
st = idxs(random_order);
if numel(st) > out_size
    st = st(1:out_size);
end

% idxs = randperm(in_size);

% st = idxs(1:out_size);

end