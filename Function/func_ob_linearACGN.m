function [y, w, R_w] = func_ob_linearACGN(A, x, SNR, R0_w, L_R0_w, flag_RC)
% Get the observation y of the signal x through the measurement equation y = Ax + w, with randomly 
% generated complex additive correlated Gaussian noise (ACGN) w~N(w; 0, R_w).
%    @y:        observation
%    @w:        measurement noise
%    @R_w:      covariance matrix of the measurement noise w
%
%    @A:        observation matrix
%    @x:        signal to be observed
%    @SNR:      signal-to-noise ratio in dB
%    @R0_w:     correlation matrix of the measurement noise w
%    @L_R0_w:   cholesky of the correlation matrix R0_w
%    @flag_RC:  flag: 'R' for real and 'C' for complex


%----------------- Function definitions ---------------------
SNR2pwr_w   = @(sig, SNR) mean(abs(sig).^2) * 10^(-SNR/10);
%------------------------------------------------------------

condif      = all( flag_RC ~= ['R', 'C'] );
if condif;  error("Argument flag_RC must be either 'R' or 'C'."); end

Ny          = size(A, 1);
Ax          = A * x;

pwr_w       = SNR2pwr_w(Ax, SNR);
R_w         = pwr_w * R0_w;
w           = sqrt(pwr_w) * L_R0_w * func_randStdN([Ny,1], flag_RC);
y           = Ax + w;

end

