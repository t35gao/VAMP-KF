classdef Leaf_IIDBernoulliGaussian
    % Leaf factor node: i.i.d. Bernoulli Gaussian (zero-mean) distribution

    properties (Constant, Access = private)
        ERRMSG_RC = "Argument flag_RC must be either 'R' or 'C'.";
    end

    properties
        flag_RC         % flag: real or complex

        rho             % sparsity (rate of zeros)
        gma             % precision of Gaussian
    end
    
    %-------------------%
    %    Constructor    %
    %-------------------%
    methods
        function obj = Leaf_IIDBernoulliGaussian(rho, gma, flag_RC)
            condif      = any( flag_RC == ['R', 'C'] );
            if condif;  obj.flag_RC = flag_RC;
            else;       error(obj.ERRMSG_RC); end
            
            obj.rho = rho;
            obj.gma = gma;
        end
    end


    %----------------------%
    %    Public Methods    %
    %----------------------%
    methods
        function s = generateRand(obj, size)
            N           = size(1) * size(2);
            sB          = zeros(N,1);                                           % Bernoulli part
            i_nz        = randperm(N, ceil(N*(1-obj.rho)));
            sB(i_nz)    = 1;
            sB          = reshape(sB, size);
            sG          = sqrt(1/obj.gma) * func_randStdN(size, obj.flag_RC);   % Gaussian part
            s           = sB.* sG;
        end

        function [mu_p, gm_p] = posteriorEst(obj, mu_e, gm_e)
            % Find the posterior (MMSE) estimate and precision given the extrinsic EP message
            %    @mu_p:     posterior estimate
            %    @gm_p:     posterior estimation-error precision
            %
            %    @mu_e:     extrinsic mean
            %    @gm_e:     extrinsic precision  

            [rhD, muN, gmN] = obj.posteriorMsg(mu_e, gm_e);
            rhN             = 1 - rhD;
            E1st            = rhN.* muN;
            E2nd            = rhN.* (1/gmN + abs(muN).^2);
            mu_p            = E1st;
            gm_p            = 1 / mean(E2nd - abs(E1st).^2);
        end
    end


    %-----------------------%
    %    Private Methods    %
    %-----------------------%
    methods (Access = private)
        function [rhD_pst, muN_pst, gmN_pst] = posteriorMsg(obj, mu_e, gm_e)
            % Find the Gaussian-mixture posterior message given the extrinsic EP message
            %    @rhD_pst:  posterior sparsity (rate of zeros)
            %    @muN_pst:  posterior Gaussian mean
            %    @gmN_pst:  posterior Gaussian precision
            %
            %    @mu_e:     extrinsic mean
            %    @gm_e:     extrinsic precision

            rho_min     = 1e-15;
            clipRhD     = @(rhD) 0*(rhD<=rho_min) + rhD.*(rhD>rho_min);

            rhN2D       = (1-obj.rho) / obj.rho;
            rhD_pst     = clipRhD(1./(1+rhN2D*obj.pdfN2N_zeroMean(mu_e,1/(1/obj.gma+1/gm_e),gm_e)));
            muN_pst     = gm_e*mu_e / (obj.gma+gm_e);
            gmN_pst     = obj.gma+gm_e;
        end

        function N2N = pdfN2N_zeroMean(obj, x, gm1, gm2)
            % Elementwise real (or complex) zero-mean Gaussian-PDF ratio, N(x;0,gm1)/N(x;0,gm2)
            %    @x:        value of "mutual" parameter x 
            %    @gm1:      numerator precision
            %    @gm2:      denominator precision
            %
            %    @N2N:      Gaussian-PDF ratio

            switch obj.flag_RC
                case 'R'; N2N = sqrt(gm1./gm2).* exp((gm2-gm1).*x.^2 / 2);
                case 'C'; N2N = (gm1./gm2).* exp((gm2-gm1).*abs(x).^2);
            end
        end
    end
end

