# VAMP-based Kalman Filtering under non-Gaussian Process Noise
##### Tiancheng Gao, Mohamed Akrout, Faouzi Bellili, and Amine Mezghani

Estimating time-varying signals becomes particularly challenging in the face of non-Gaussian (e.g., sparse) and/or rapidly time-varying process noise. By building upon the recent progress in the approximate message passing (AMP) paradigm, this paper unifies the vector variant of AMP (i.e., VAMP) with the Kalman filter (KF) into a unified message passing framework. The new algorithm (coined VAMP-KF) does not restrict the process noise to a specific structure (e.g., same support over time), thereby accounting for non-Gaussian process noise sources that are uncorrelated both component-wise and over time. For the sake of theoretical performance prediction, we conduct a state evolution (SE) analysis of the proposed algorithm and show its consistency with the asymptotic empirical mean-squared error (MSE). Numerical results using sparse noise dynamics with different sparsity ratios demonstrate unambiguously the effectiveness of the proposed VAMP-KF algorithm and its superiority over state-of-the-art algorithms both in terms of reconstruction accuracy and computational complexity.

## Running experiments
### Description of main files
| Script &nbsp; &nbsp; &nbsp; &nbsp; | Output |
| :---         |     :---      |
| main_heatmap_TNRMSEvsAlpnRho_BG.m     |     TNRMSE vs alpha and rho for a Bernoulli-Gaussian process noise            |
| main_heatmap_TNRMSEvsM2NnRho_BG.m     |     TNRMSE vs M/N and rho for a Bernoulli-Gaussian process noise (Fig. 12)    |
| main_heatmap_TNRMSEvsM2NnSNR_BG.m     |     TNRMSE vs M/N and SNR for a Bernoulli-Gaussian process noise (Fig. 11)    |
| main_heatmap_TNRMSEvsM2NnSNR_binary.m |     TNRMSE vs M/N and SNR for a binary process noise (Fig. 14)                |
| main_TNRMSEvsAlp_BG.m                 |     TNRMSE vs alpha for a Bernoulli-Gaussian process noise (Fig. 10)          |
| main_TNRMSEvsM2N_BG.m                 |     TNRMSE vs M/N for a Bernoulli-Gaussian process noise (Fig. 5)             |
| main_TNRMSEvsM2N_binary.m             |     TNRMSE vs M/N for a binary process noise (Fig. 13)                        |
| main_TNRMSEvsRho_BG.m                 |     TNRMSE vs rho for a Bernoulli-Gaussian process noise (Fig. 9)             |
| main_TNRMSEvsSNR_BG.m                 |     TNRMSE vs SNR for a Bernoulli-Gaussian process noise (Fig. 6)             |
| main_TNRMSEvsSNR_binary.m             |     TNRMSE vs SNR for a binary process noise                                  |

### Note
- For all our Monte-Carlo simulations, we fix the seed in all scripts for reproducibility purposes. Feel free to change or remove it accordingly.
- Some scripts include an interactive input prompt requesting the values of additional parameters.
