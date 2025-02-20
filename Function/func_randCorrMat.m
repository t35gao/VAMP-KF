function [R0, L_R0] = func_randCorrMat(N, flag_RC)
% Generate a real (or complex) random correlation matrix.
%   @R0:        the correlation matrix
%   @L_R0:      cholesky of R0 (i.e., R0 = L_R0' * L_R0)
%
%   @N:         size of the correlation matrix
%   @flag_RC:   flag: 'R' for real and 'C' for complex

condif      = all( flag_RC ~= ['R', 'C'] );
if condif;  error("Argument flag_RC must be either 'R' or 'C'."); end

[Q, ~]  = qr( func_randStdN([N,N], flag_RC) );
buffer  = func_randStdN([N,N], flag_RC);
R0      = corr([Q; buffer]);
L_R0    = chol(R0)';
end

