classdef LeafOb_linearAWGN
    % Leaf factor node: observaton y = Ax + w
    %   x: signal to be estimated
    %   A: linear observation matrix
    %   w: additive white Gaussian noise (AWGN)
    
    properties
        % SVD-by-rank of the observation matrix A with rank R
        U_SVD       % matrix: U'*U = eye(R)
        V_SVD       % matrix: V'*V = eye(R)
        s_SVD       % vector: length(s) = R
        
        y           % observation
        gm_w        % noise precision
    end
    
    methods
        function obj = LeafOb_linearAWGN(SVD_A, y, gm_w)
            obj.U_SVD       = SVD_A.U;
            obj.V_SVD       = SVD_A.V;
            obj.s_SVD       = SVD_A.s;
            obj.y           = y;
            obj.gm_w        = gm_w;
        end
        
        function [mu_p, gm_p] = posteriorEst(obj, mu_e, gm_e)
            % Find the posterior (MMSE) estimate and precision given the extrinsic EP message
            %    @mu_p:     posterior estimate
            %    @gm_p:     posterior estimation-error precision
            %
            %    @mu_e:     extrinsic mean
            %    @gm_e:     extrinsic precision  

            U           = obj.U_SVD;
            V           = obj.V_SVD;
            s           = obj.s_SVD;

            [N,R]       = size(V);
            s_square    = abs(s).^2;
            zTld        = U' * (obj.y)./ s;
            
            d           = s_square./ (s_square + gm_e/obj.gm_w);
            mu_p        = mu_e + V * (d.* (zTld - V'*mu_e));
            gm_p        = gm_e / (1 - (R/N)*mean(d));
        end
    end
end

