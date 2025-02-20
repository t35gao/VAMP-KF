function [xEst, PEst] = MCCKF(F, H, y, Q, R, xEst, PEst)
    Nx          = size(xEst, 1);

    xEst        = F * xEst;
    PPred       = F * PEst * F' + Q;
    invers_R    = pinv(R);
    innov       = y - H * xEst;
    norm_innov  = sqrt((innov)' * invers_R * (innov));
    sigma       = 1 * norm_innov;
    K           = exp(-(norm_innov^2) /(2 * sigma^2));
    Gain        = pinv(pinv(PPred) + K * H' * invers_R * H) * K * (H') * invers_R;
    xEst        = xEst + Gain *(innov);
    PEst        = (eye(Nx) - Gain*H) * PPred * (eye(Nx) - Gain*H)' + Gain * R * Gain';
end