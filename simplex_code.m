function C = simplex_code(K)

if K == 1
    error('K must be >= 2');
end

C = [1 -1];
for i = 2:K-1
    u = -ones(i,1) / i;
    v = zeros(i-1, 1);
    mult_factor = sqrt(1 - (1/(i^2)));
    C = [1       u';
         v C*mult_factor];
end