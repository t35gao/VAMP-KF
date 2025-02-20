function [gma_x_est] = algo_SE3_VAMPKF(sysPkg_SE, gma_w, leaf_v1, itPrams)

% unpackaging
L2M             = sysPkg_SE.L2M;
N2M             = sysPkg_SE.N2M;
LvarG           = sysPkg_SE.LvarG;
MvarH           = sysPkg_SE.MvarH;
NvarF           = sysPkg_SE.NvarF;
NvarG           = sysPkg_SE.NvarG;
NvarH           = sysPkg_SE.NvarH;
gma_x_est_old   = sysPkg_SE.gma_x_est_old;

% pseudo noise precision gma_n
gma_n           = 1 / ( NvarF*NvarH/gma_x_est_old + 1/gma_w );


%% VAMP SE
MvarA           = MvarH * NvarG;
[~, gma_v2e]    = algo_SE1_VAMP(L2M, MvarA, gma_n, leaf_v1, itPrams);


%% KF SE
gma_x_est       = algo_SE2_KF(N2M, LvarG, MvarH, NvarF, ...
                              gma_v2e, gma_w, gma_x_est_old);


end