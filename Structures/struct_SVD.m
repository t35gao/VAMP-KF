function [svd_A] = struct_SVD(A)
[U, S, V] = svd(A, 'econ');
R = rank(A);
U = U(:,1:R);
V = V(:,1:R);  
s = diag(S(1:R,1:R));

svd_A = struct('mat', A, ...
               'U', U, ...
               'V', V, ...
               's', s);
end