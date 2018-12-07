function K = rbf(X1, X2, params)

% K = zeros(size(X1,1), size(X2,1));
% for i = 1:size(X1,1)
%     
%     dx = zeros(size(X2,1), size(X1,2));
%     for j = 1:size(X2,1)
%         dx(j,:) = X1(i,:) - X2(j,:);
%     end
%     
% %     keyboard
%     
%     K(i,:) = fastexp(-params.gamma*sum(dx.^2,2));
% end

sum1 = sum(X1.^2,2);
sum2 = sum(X2.^2,2);
K = exp(-params.gamma*(sum1*ones(1,size(X2,1)) + ones(size(X1,1),1)*sum2' - 2*X1*X2'));

% function ex = fastexp(x)
% 
% ex = 1.0 + x / (8*4096);
% for i = 1:15
%     ex = ex.*ex;
% end