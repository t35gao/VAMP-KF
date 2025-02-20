function [x_est, S_est] = algo_2_KF(sysPkg, y, R_w, m_v, C_v)

% Unpackaging
F           = sysPkg.F;
G           = sysPkg.G;
H           = sysPkg.H;
x_est_old   = sysPkg.x_est_old;
S_est_old   = sysPkg.Sgm_est_old;

Nx        	= length(x_est_old);


%% Prediction Step
x_prd     	= F * x_est_old + G * m_v;
S_prd       = F * S_est_old * F' + G * C_v * G';

% getting rid of asymmetry and complex diag due to finite precision
S_prd       = (S_prd + S_prd') / 2;


%% Update Step
HSp      	= H * S_prd;
K           = HSp' / ( H * HSp' + R_w );

KH          = K * H;
x_est       = x_prd + K * y - KH * x_prd;
S_est       = ( eye(Nx) - KH ) * S_prd;

% getting rid of asymmetry and complex diag due to finite precision
S_est       = (S_est + S_est') / 2;


end