classdef Leaf_IIDCplxBinary
    % Leaf factor node: i.i.d. complex binary, i.e., 
    % discrete uniform distribution over the 4 points {a+1i*b, a-1i*b, -a+1i*b, -a-1i*b}
    properties (Constant)
        flag_RC     = 'C'
        gm_max      = 1e50;
    end

    properties
        a
        b
    end
    
    methods
        function obj = Leaf_IIDCplxBinary(a, b)
            obj.a = a;
            obj.b = b;
        end

        function s = generateRand(obj, size)
            N           = size(1) * size(2);
            i_re        = randperm(N, ceil(N/2));
            s_re        = obj.a * ones(N,1);
            s_re(i_re)  = obj.b;

            i_im        = randperm(N, ceil(N/2));
            s_im        = obj.a * ones(N,1);
            s_im(i_im)  = obj.b;

            s           = reshape(s_re + 1i*s_im, size);
        end
        
        function [mu_p, gm_p] = posteriorEst(obj, mu_e, gm_e)
            % Find the posterior (MMSE) estimate and precision given the extrinsic EP message
            %    @mu_p:     posterior estimate
            %    @gm_p:     posterior estimation-error precision
            %
            %    @mu_e:     extrinsic mean
            %    @gm_e:     extrinsic precision

            gmClip      = @(gm) min(gm, obj.gm_max);

            N           = length(mu_e);

            m           = (obj.a + obj.b) / 2;
            d           = (obj.a - obj.b) / 2;

            m_e_ccat    = [real(mu_e); imag(mu_e)];
            gma_e_ccat  = 2 * gm_e;

            term        = gma_e_ccat * d * (m_e_ccat - m);
            mu_p_ccat   = m + d * tanh(term);
            gm_p_ccat   = 1 / mean( d^2 * sech(term).^2 );

            mu_p        = mu_p_ccat(1:N) + 1i*mu_p_ccat(N+1:end);
            gm_p        = gmClip(gm_p_ccat/2);
        end  
    end
end

