classdef SysPkg_SE
    % The package contains the asymptotic dimension ratios as well as 
    % gma_x_est_old that are used in the state evolution (SE) analysis of 
    % VAMP-KF.
    
    properties
        L2M
        N2M

        LvarG
        MvarH
        NvarF
        NvarG
        NvarH

        gma_x_est_old
    end
    
    methods
        function obj = SysPkg_SE(L2M, N2M, ...
                                 LvarG, MvarH, NvarF, NvarG, NvarH, ...
                                 gma_x_est_old)
            obj.L2M             = L2M;
            obj.N2M             = N2M;

            obj.LvarG           = LvarG;
            obj.MvarH           = MvarH;
            obj.NvarF           = NvarF;
            obj.NvarG           = NvarG;
            obj.NvarH           = NvarH;

            obj.gma_x_est_old   = gma_x_est_old;
        end

        function obj = updateOld(obj, gma_x_est_old)
            obj.gma_x_est_old   = gma_x_est_old;
        end
    end
end

