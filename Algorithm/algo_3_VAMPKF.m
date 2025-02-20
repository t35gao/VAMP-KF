function [x_est, S_est] = algo_3_VAMPKF(sysPkg, y, R_w, leaf_v1, itPrams)

% unpackaging
x_est_old   = sysPkg.x_est_old;
S_est_old   = sysPkg.Sgm_est_old;
HF          = sysPkg.HF;
SVD_A       = sysPkg.SVD_HG;

% pseudo observation n
% pseudo noise covariance Sgm_n and precision gma_n
n           = y - HF * x_est_old;
Sgm_n       = HF * S_est_old * HF' + R_w;
Sgm_n       = (Sgm_n + Sgm_n')/2;
gma_n       = 1 / mean(diag(Sgm_n));

% pseudo leaf factor node on v- (i.e., likelihood p(n|v-))
leaf_v2     = LeafOb_linearAWGN(SVD_A, n, gma_n);


%% VAMP
[~, ~, v2e, gma_v2e] ...
            = algo_1_VAMP_SVD(leaf_v1, leaf_v2, itPrams);


%% KF
Nv          = size(v2e, 1);
Sgm_v2e     = (1/gma_v2e)*eye(Nv);

[x_est, S_est] ...
            = algo_2_KF(sysPkg, y, R_w, v2e, Sgm_v2e);


end

