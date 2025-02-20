function [gm_v1p_, gm_v2e_] = algo_SE1_VAMP(L2M, MvarA, gma_n, leaf_v1, itPrams)

%----------------- Function definitions ---------------------
funcF       = @(x, z) ( sqrt(x*(1+sqrt(z))^2+1) - ...
                        sqrt(x*(1-sqrt(z))^2+1) )^2;
getMSE      = @(xhat, x) mean(abs(xhat-x).^2);
%------------------------------------------------------------

K_it        = itPrams.K_it;

N_sample    = 1e4;
v0          = leaf_v1.generateRand([N_sample,1]);

gm_v1p      = zeros(1, K_it);
gm_v2e      = zeros(1, K_it);
gm_v2p      = zeros(1, K_it);
gm_v1e      = zeros(1, K_it+1);

gm_v1e(1)   = itPrams.gm_v1e_init;

for k = 1:K_it
    %% MMSE
    v1e         = v0 + sqrt(1/gm_v1e(k)) * func_randStdN([N_sample,1], leaf_v1.flag_RC);
    [v1p, ~]    = leaf_v1.posteriorEst(v1e, gm_v1e(k));
    gm_v1p(k)   = 1 / getMSE(v1p, v0);
    gm_v2e(k)   = gm_v1p(k) - gm_v1e(k);
    
    
    %% LMMSE
    termF       = funcF( MvarA * gma_n / gm_v2e(k), L2M );
    gm_v2p(k)   = 1 / ( 1/gm_v2e(k) - termF/(4*L2M*MvarA*gma_n) );
    gm_v1e(k+1) = gm_v2p(k) - gm_v2e(k);
    
end

gm_v1p_ = gm_v1p(end);
gm_v2e_ = gm_v2e(end);
end