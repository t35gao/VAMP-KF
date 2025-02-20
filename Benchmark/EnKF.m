function [xEst, X_enkf] = EnKF(F, H, y, Q, R, X_enkf, N_Enkf)
    X_enkf = F * X_enkf;
    no     = sqrt(diag(diag(Q))) * randn(size(X_enkf));
    X_enkf = X_enkf + no;
    xEst   = mean(X_enkf, 2);
    PPred  = (1/(N_Enkf-1)) * ...
                ((X_enkf - repmat(xEst, 1, N_Enkf))*(X_enkf - repmat(xEst, 1, N_Enkf)).');
    K      = PPred * H' / (H * PPred * H' + R);
    X_enkf = X_enkf + K * (repmat(y, 1, N_Enkf) - H * X_enkf);
    xEst   = mean(X_enkf, 2);
    %{
    PEst   = (1/(N_Enkf-1)) * ...
                ((X_enkf - repmat(xEst, 1, N_Enkf))*(X_enkf - repmat(xEst, 1, N_Enkf)).');
    %}
end