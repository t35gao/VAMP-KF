function [y, w, gm_w] = func_ob_linearAWGN(A, x, SNR, flag_RC)
% Get the observation y of the signal x through the measurement equation y = Ax + w, with randomly 
% generated complex additive white Gaussian noise (AWGN) w~N(w; 0, 1/gm_w).
%    @y:        observation
%    @w:        measurement noise
%    @gm_w:     precision of the measurement noise w
%
%    @A:        observation matrix
%    @x:        signal to be observed
%    @SNR:      signal-to-noise ratio in dB
%    @flag_RC:  flag: 'R' for real and 'C' for complex


%----------------- Function definitions ---------------------
SNR2pwr_w   = @(sig, SNR) mean(abs(sig).^2) * 10^(-SNR/10);
%------------------------------------------------------------

condif      = all( flag_RC ~= ['R', 'C'] );
if condif;  error("Argument flag_RC must be either 'R' or 'C'."); end

Ny          = size(A, 1);
Ax          = A * x;

pwr_w       = SNR2pwr_w(Ax, SNR);
gm_w        = 1 / pwr_w;
w           = sqrt(pwr_w) * func_randStdN([Ny,1], flag_RC);
y           = Ax + w;

end

