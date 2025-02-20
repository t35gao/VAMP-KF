
config_main

rng(100);

%----------------- Function definitions ---------------------
getMSE      = @(xhat, x) mean(abs(xhat-x).^2);
getNRMSE    = @(xhat, x) sqrt( mean(abs(xhat-x).^2) / ...
                               mean(abs(x).^2) );
getPower    = @(x) mean(abs(x).^2);
%------------------------------------------------------------



%% Simulation Parameters
% flag: real or complex
flag_RC         = 'C';

% SNR
SNR_options     = 5:5:30;

% system dimension
Nx              = 200;
Nv              = Nx;
Ny_options      = 20:20:Nx*2;

% counters
t_warmup     	= 20;                   % number of warm-up time steps
N_time          = 100 + t_warmup;       % number of total time steps
N_MC            = 100;                  % number of Monte Carlo tests

% iterative-method parameters for VAMP
gm_v1e_init     = 1e-4;
v1e_init        = sqrt(1/gm_v1e_init) * func_randStdN([Nv,1], flag_RC);
K_iter          = 500;
thold_break     = 1e-4;
ratio_damp      = 1;
itPrams         = struct_VAMPIterParams(v1e_init, gm_v1e_init, K_iter, thold_break, ratio_damp);

% process scaling factor alpha
alp             = 0.7;

% innovation parameters
power_v         = Nx/Nv;                % to get power(x) = power(Gv) = 1
gma_v           = 1/power_v;            % percision of innovation v
rho_BCN         = 0.9;                  % sparsity rho (ratio of zeros)
gma_BCN         = (1-rho_BCN)*gma_v;    % precision of the Gaussian
% leaf factor node on v+ (i.e., prior p(v+))
leaf_v1         = Leaf_IIDBernoulliGaussian(rho_BCN, gma_BCN, flag_RC);

% NRMSE storage matrices
size_storage    = [length(SNR_options), length(Ny_options), N_MC, N_time];
NRMSE_x_VAMPKF 	= zeros(size_storage);



%% Simulation Implementation
xht_init   	= zeros(Nx, 1);
Sht_init  	= eye(Nx);

for i_SNR = 1:length(SNR_options)
    SNR     = SNR_options(i_SNR);
    
    for i_M2N = 1:length(Ny_options)
        Ny  = Ny_options(i_M2N);
        M2N = Ny/Nx;
        

        %----------------------------%
        %        Monte Carlos        %
        %----------------------------%
        for i_MC = 1:N_MC
            % state transition matrix F
            % process noise transformation matrix G
            % observation matrix H
            varF            = alp^2     / Nx;
            varG            = (1-alp^2) / Nx;
            varH            = 1 / Ny;
            F               = sqrt(varF) * func_randStdN([Nx,Nx], flag_RC);
            G               = sqrt(varG) * func_randStdN([Nx,Nv], flag_RC);
            H               = sqrt(varH) * func_randStdN([Ny,Nx], flag_RC);
            % measurement noise correlation R0_w and its cholesky L_R0_w (i.e., R0 = L_R0' * L_R0)
            [R0_w, L_R0_w]  = func_randCorrMat(Ny, flag_RC);
            % packaging algorithm inputs
            sysPkg_KF       = SysPkg(F, G, H, xht_init, Sht_init);
            sysPkg_VAMPKF   = SysPkg(F, G, H, xht_init, Sht_init);

            % signal x at t = 0 (unit-power)
            power_x         = 1;
            x               = sqrt(power_x) * func_randStdN([Nx,1], flag_RC);

            % VAMP-KF at t = 0
            xht_old_VAMPKF  = xht_init;
            Sht_old_VAMPKF	= Sht_init;
            

            %----------------------------%
            %    Temporal Evolutions     %
            %----------------------------%
            for t = 1:N_time
                % innovation v, signal x, observation y
                % measurement noise w, noise covariance R_w
                v   	    = leaf_v1.generateRand([Nv,1]);
                x      	    = F * x + G * v;
                [y, w, R_w] = func_ob_linearACGN(H, x, SNR, R0_w, L_R0_w, flag_RC);
                
                
                % --------- VAMP-KF ----------
                [xht_VAMPKF, Sht_VAMPKF] = algo_3_VAMPKF(sysPkg_VAMPKF, y, R_w, leaf_v1, itPrams);
                NRMSE_x_VAMPKF(i_SNR, i_M2N, i_MC, t) = getNRMSE(xht_VAMPKF, x);
                % -------- update old --------
                xht_old_VAMPKF  = xht_VAMPKF;
                Sht_old_VAMPKF  = Sht_VAMPKF;
                sysPkg_VAMPKF   = sysPkg_VAMPKF.updateOld(xht_old_VAMPKF, Sht_old_VAMPKF);
                
                
                % displaying NRMSE
                fprintf(1, "[SNR = %02d][M2N = %0.2f][MC = %03d][t = %03d]: ", SNR, M2N, i_MC, t);
                fprintf(1, "e_VAMP-KF = %8.5f\n", NRMSE_x_VAMPKF(i_SNR, i_M2N, i_MC, t));
            end
        end
    end
end



%% Simulation Results
TNRMSE_x_VAMPKF = squeeze(mean(NRMSE_x_VAMPKF(:,:,t_warmup+1:end), [3 4]));
imagesc(Ny_options/Nx, SNR_options, TNRMSE_x_VAMPKF);
colormap( flipud(colormap('Gray')) ); 
colorbar

xlabel('$M/N$', 'interpreter', 'latex')
ylabel('SNR', 'interpreter', 'latex')


fig_pos     = [10, 80, 900, 300];
[cf, ca]    = config_fig(fig_pos);

ca.YDir                 = 'normal';

cb                      = ca.Colorbar;
cb.Label.String         = 'TNRMSE';
cb.Label.Interpreter    = 'latex';
cb.TickDirection        = 'out';
cb.Limits               = [0 1];
cb.FontSize             = 20;
