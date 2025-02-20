classdef SysPkg
    % The package contains the transition matrices F, G, H, there variants, 
    % and the old estimate x_est_old with covariance Sgm_est_old, which are
    % used as the input parameters of the algorithms.
    
    properties
        F
        G
        H

        HF          % H*F
        SVD_HG      % SVD-by-rank of H*F

        x_est_old
        Sgm_est_old
    end
    
    methods
        function obj = SysPkg(F, G, H, x_est_old, Sgm_est_old)
            obj.F           = F;
            obj.G           = G;
            obj.H           = H;
            
            obj.HF          = H * F;
            obj.SVD_HG      = struct_SVD(H * G);

            obj.x_est_old   = x_est_old;
            obj.Sgm_est_old = Sgm_est_old;
        end
        
        function obj = updateOld(obj, x_est_old, Sgm_est_old)
            obj.x_est_old   = x_est_old;
            obj.Sgm_est_old = Sgm_est_old;
        end
    end
end

