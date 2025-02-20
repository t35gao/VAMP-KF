
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
Ny              = 100;

% counters
t_warmup     	= 0;                    % number of warm-up time steps
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
% leaf factor node on v+ (i.e., prior p(v+))
symbol_B        = [0, 1];
factor_scale    = sqrt(power_v / (2*getPower(symbol_B)));
symbol_CB       = factor_scale * symbol_B;
leaf_v1         = Leaf_IIDCplxBinary(symbol_CB(1), symbol_CB(2));

% NRMSE storage matrices
size_storage    = [length(SNR_options), N_MC, N_time];
NRMSE_x_KF     	= zeros(size_storage);
NRMSE_x_EnKF   	= zeros(size_storage);
NRMSE_x_MCCKF 	= zeros(size_storage);
NRMSE_x_VAMPKF  = zeros(size_storage);



%% Simulation Implementation
xht_init   	= zeros(Nx, 1);
Sht_init  	= eye(Nx);

for i_SNR = 1:length(SNR_options)
    SNR     = SNR_options(i_SNR);
    

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
        % covariance of Gv (used in benchmarks)
        Q_Gv            = G*G'/gma_v;
        % measurement noise correlation R0_w and its cholesky L_R0_w (i.e., R0 = L_R0' * L_R0)
        [R0_w, L_R0_w]  = func_randCorrMat(Ny, flag_RC);
        % packaging algorithm inputs
        sysPkg_KF       = SysPkg(F, G, H, xht_init, Sht_init);
        sysPkg_VAMPKF   = SysPkg(F, G, H, xht_init, Sht_init);

        % signal x at t = 0 (unit-power)
        power_x         = 1;
        x               = sqrt(power_x) * func_randStdN([Nx,1], flag_RC);

        % benchmarks at t = 0
        % ------------ KF ------------
        xht_old_KF     	= xht_init;
        Sht_old_KF    	= Sht_init;
        % ----------- EnKF -----------
        xht_old_EnKF   	= xht_init;
        Sht_old_EnKF  	= Sht_init;
        N_EnKF          = 2000;
        no              = sqrt(Sht_old_EnKF) * repmat(func_randStdN([Nx,1], flag_RC), 1, N_EnKF);
        X_old_EnKF      = repmat( xht_old_EnKF, 1, N_EnKF ) + no;
        % ---------- MCC-KF ----------
        xht_old_MCCKF   = xht_init;
        Sht_old_MCCKF	= Sht_init;
        % --------- VAMP-KF ----------
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
            
            
            %----------------------------%
            %         Benchmarks         %
            %----------------------------%
            % ------------ KF ------------
            [xht_KF, Sht_KF] = algo_2_KF(sysPkg_KF, y, R_w, zeros(Nv,1), eye(Nv)/gma_v);
            NRMSE_x_KF(i_SNR, i_MC, t) = getNRMSE(xht_KF, x);
            % ----------- EnKF -----------
            [xht_EnKF, X_EnKF] = EnKF(F, H, y, Q_Gv, R_w, X_old_EnKF, N_EnKF);
            NRMSE_x_EnKF(i_SNR, i_MC, t) = getNRMSE(xht_EnKF, x);
            % ---------- MCC-KF ----------
            [xht_MCCKF, Sht_MCCKF] = MCCKF(F, H, y, Q_Gv, R_w, xht_old_MCCKF, Sht_old_MCCKF);
            NRMSE_x_MCCKF(i_SNR, i_MC, t) = getNRMSE(xht_MCCKF, x);
            % --------- VAMP-KF ----------
            if SNR>=25; [~,~,R_w] = func_ob_linearCplxACGN(H, x, SNR-5, R0_w, L_R0_w, flag_RC); end
            [xht_VAMPKF, Sht_VAMPKF] = algo_3_VAMPKF(sysPkg_VAMPKF, y, R_w, leaf_v1, itPrams);
            NRMSE_x_VAMPKF(i_SNR, i_MC, t) = getNRMSE(xht_VAMPKF, x);
            % -------- update old --------
            xht_old_KF      = xht_KF;
            Sht_old_KF      = Sht_KF;
            X_old_EnKF      = X_EnKF;
            xht_old_MCCKF   = xht_MCCKF;
            Sht_old_MCCKF   = Sht_MCCKF;
            xht_old_VAMPKF  = xht_VAMPKF;
            Sht_old_VAMPKF  = Sht_VAMPKF;
            sysPkg_KF       = sysPkg_KF.updateOld(xht_old_KF, Sht_old_KF);
            sysPkg_VAMPKF   = sysPkg_VAMPKF.updateOld(xht_old_VAMPKF, Sht_old_VAMPKF);
            
            
            % displaying NRMSE
            fprintf(1, '[SNR = %02d][MC = %03d][t = %03d]: ', SNR, i_MC, t);
            fprintf(1, "e_KF = %8.5f"      + ", " + ...
                       "e_EnKF = %8.5f"    + ", " + ...
                       "e_MCC-KF = %8.5f"  + ", " + ...
                       "e_VAMP-KF = %8.5f" + "\n" , ...
                       NRMSE_x_KF(i_SNR, i_MC, t), ...
                       NRMSE_x_EnKF(i_SNR, i_MC, t), ...
                       NRMSE_x_MCCKF(i_SNR, i_MC, t), ...
                       NRMSE_x_VAMPKF(i_SNR, i_MC, t));
        end
    end
end



%% Simulation Results
% HandleVisibility off: prevent error bar from appearing in legend
colors = func_distinguishableColors(5);

sqrt_N_sample = sqrt(N_MC * N_time);

TNRMSE_x_KF = squeeze(mean(NRMSE_x_KF(:,:,t_warmup+1:end), [2 3]));
TNRMSE_x_KF_std = squeeze(std(NRMSE_x_KF(:,:,t_warmup+1:end), 1, [2 3]));
TNRMSE_x_KF_ste = TNRMSE_x_KF_std / sqrt_N_sample;
semilogy(SNR_options, TNRMSE_x_KF, 'DisplayName', 'standard KF', ...
    'LineWidth', 3, 'Color', colors(4,:)); hold on;
errorbar(SNR_options, TNRMSE_x_KF, TNRMSE_x_KF_ste/2, 'o', ...
    'LineWidth', 2, 'Color', colors(4,:), 'HandleVisibility', 'off'); hold on;

TNRMSE_x_EnKF = squeeze(mean(NRMSE_x_EnKF(:,:,t_warmup+1:end), [2 3]));
TNRMSE_x_EnKF_std = squeeze(std(NRMSE_x_EnKF(:,:,t_warmup+1:end), 0, [2 3]));
TNRMSE_x_EnKF_ste = TNRMSE_x_EnKF_std / sqrt_N_sample;
semilogy(SNR_options, TNRMSE_x_EnKF, 'DisplayName', 'EnKF', ...
    'LineWidth', 3, 'Color', colors(3,:)); hold on;
errorbar(SNR_options, TNRMSE_x_EnKF, TNRMSE_x_EnKF_ste/2, 'o', ...
    'LineWidth', 2, 'Color', colors(3,:), 'HandleVisibility', 'off'); hold on;

TNRMSE_x_MCCKF = squeeze(mean(NRMSE_x_MCCKF(:,:,t_warmup+1:end), [2 3]));
TNRMSE_x_MCCKF_std = squeeze(std(NRMSE_x_MCCKF(:,:,t_warmup+1:end), 0, [2 3]))/2;
TNRMSE_x_MCCKF_ste = TNRMSE_x_MCCKF_std / sqrt_N_sample;
semilogy(SNR_options, TNRMSE_x_MCCKF, 'DisplayName', 'MCC-KF', ...
    'LineWidth', 3, 'Color', colors(2,:)); hold on;
errorbar(SNR_options, TNRMSE_x_MCCKF, TNRMSE_x_MCCKF_ste/2, 'o', ...
    'LineWidth', 2, 'Color', colors(2,:), 'HandleVisibility', 'off'); hold on;

TNRMSE_x_VAMPKF = squeeze(mean(NRMSE_x_VAMPKF(:,:,t_warmup+1:end), [2 3]));
TNRMSE_x_VAMPKF_std = squeeze(std(NRMSE_x_VAMPKF(:,:,t_warmup+1:end), 0, [2 3]));
TNRMSE_x_VAMPKF_ste = TNRMSE_x_VAMPKF_std / sqrt_N_sample;
semilogy(SNR_options, TNRMSE_x_VAMPKF, 'DisplayName', 'VAMP-KF', ...
    'LineWidth', 3, 'Color', colors(1,:)); hold on;
errorbar(SNR_options, TNRMSE_x_VAMPKF, TNRMSE_x_VAMPKF_ste/2, 'o', ...
    'LineWidth', 2, 'Color', colors(1,:), 'HandleVisibility', 'off'); hold on;


%{
TNRMSE_x_SE = squeeze(mean(NRMSE_x_SE(:,t_warmup+1:end), 2));
TNRMSE_x_SE_std = squeeze(std(NRMSE_x_SE(:,t_warmup+1:end), 0, 2));
TNRMSE_x_SE_ste = TNRMSE_x_SE_std / N_time;
semilogy(SNR_options, TNRMSE_x_SE, '--', 'DisplayName', 'VAMP-KF SE', ...
    'LineWidth', 3, 'Color', colors(5,:)); hold on;
errorbar(SNR_options, TNRMSE_x_SE, TNRMSE_x_SE_ste/2, '--o', ...
    'LineWidth', 2, 'Color', colors(5,:), 'HandleVisibility', 'off'); hold on;
%}

hold off

xlabel('SNR', 'interpreter', 'latex')
ylabel('TNRMSE', 'interpreter', 'latex')

legend('Location', 'southwest', 'fontsize', 24)
legend show
grid on


fig_pos     = [10, 80, 730, 700];
[cf, ca]    = config_fig(fig_pos);
