function [gma_x_est] = algo_SE2_KF(N2M, LvarG, MvarH, NvarF, ...
                                   gma_v2e, gma_w, gma_x_est_old)

%----------------- Function definitions ---------------------
funcF       = @(x, z) ( sqrt(x*(1+sqrt(z))^2+1) - ...
                        sqrt(x*(1-sqrt(z))^2+1) )^2;
%------------------------------------------------------------


%% Prediction
gma_x_prd	= 1 / ( NvarF/gma_x_est_old + LvarG/gma_v2e ); 


%% Update
termF       = funcF( MvarH * gma_w / gma_x_prd, N2M );
gma_x_est 	= 1 / ( 1/gma_x_prd - termF/(4*N2M*MvarH*gma_w) );


end