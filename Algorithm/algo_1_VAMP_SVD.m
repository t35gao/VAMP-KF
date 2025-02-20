function [v1p_, gm_v1p_, v2e_, gm_v2e_] = algo_1_VAMP_SVD(leaf_v1, leaf_v2, itPrams)

%----------------- Function definitions ---------------------
damp        = @(r, x, x_old) r*x + (1-r)*x_old;
gmClip      = @(gm) max(min(gm, 1e11), 1e-11);
getNRMSE    = @(xhat, x) sqrt( mean(abs(xhat-x).^2) / mean(abs(x).^2) );
%------------------------------------------------------------

K_it        = itPrams.K_it;

Nv        	= size(leaf_v2.V_SVD, 1);

v1p         = zeros(Nv, K_it);
v2e         = zeros(Nv, K_it);
v2p     	= zeros(Nv, K_it);
v1e     	= zeros(Nv, K_it+1);

gm_v1p      = zeros(1, K_it);
gm_v2e      = zeros(1, K_it);
gm_v2p	    = zeros(1, K_it);
gm_v1e	    = zeros(1, K_it+1);

v1e(:,1)    = itPrams.v1e_init;
gm_v1e(1)   = itPrams.gm_v1e_init;

v1p_old     = zeros(Nv, 1);

k_end       = K_it;


for k = 1:K_it
    r_damp      = (k==1) * 1 + (k>1) * itPrams.r_dmp;

    
    %% Estimate
    % posterior estimate of v+
    [v1p_raw, gm_v1p_raw] ...
                = leaf_v1.posteriorEst(v1e(:,k), gm_v1e(k));
    v1p(:,k)    = damp(r_damp, v1p_raw, v1p_old);
    v1p_old     = v1p(:,k);
    gm_v1p(k)   = gm_v1p_raw;
    
    % extrinsic estimate of v-
    gm_v2e_raw  = gm_v1p(k) - gm_v1e(k);
    gm_v2e(k)   = gmClip(gm_v2e_raw);
    v2e(:,k)    = (gm_v1p(k) * v1p(:,k) - gm_v1e(k) * v1e(:,k)) / gm_v2e_raw;
    
    % posterior estimate of v-
    [v2p_raw, gm_v2p_raw] ...
                = leaf_v2.posteriorEst(v2e(:,k), gm_v2e(k));
    v2p(:,k)    = v2p_raw;
    gm_v2p(k)   = gm_v2p_raw;
    
    % extrinsic estimate of v+
    gm_v1e_raw  = gm_v2p(k) - gm_v2e(k);
    gm_v1e(k+1) = gmClip(gm_v1e_raw);
    gm_v1e(k+1) = damp(r_damp, gm_v1e(k+1), gm_v1e(k));
    v1e(:,k+1)  = (gm_v2p(k) * v2p(:,k) - gm_v2e(k) * v2e(:,k)) / gm_v1e_raw;


    %% Check convergence
    diff_v1e        = getNRMSE(v1e(:,k), v1e(:,k+1));
    if diff_v1e < itPrams.thd_brk
        k_end = k;  break;
    end
    
end

v1p_        = v1p(:,k_end);
v2e_        = v2e(:,k_end);
gm_v1p_     = gm_v1p(k_end);
gm_v2e_     = gm_v2e(k_end);
end

