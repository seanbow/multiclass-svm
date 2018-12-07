function K = polynomial(X1, X2, params)

% if nargin <= 2
%     d = 2;
% end


% m1 = size(X1, 1);
% m2 = size(X2, 1);
% 
% K = zeros(m1, m2);

% Normalize the kernel so that dot products within [0,1] produce kernel
% values in [0,1]
norm_factor = 1 / (2^params.d);

% norm_factor = norm_factor^2; % ???

K = norm_factor * (X1*X2' + 1).^params.d;

% for i = 1:m1
% %     for j = 1:m2
% %         K(i,j) = norm_factor * (X1(i,:) * X2(j,:)' + 1).^params.d;
% %     end
% end