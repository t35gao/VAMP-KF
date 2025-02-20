function itPrams = struct_VAMPIterParams(v1e_init, gm_v1e_init, K_it, thd_brk, r_dmp)
% Generate a structure of iterative-method parameters for VAMP.
%   @v1e_init:      initial value for v1e
%   @gm_v1e_init:   initial value for gm_v1e
%   @K_it:          maximum number of iterations
%   @thd_brk:       threshold (diff(v1e)) below which the loop breaks
%   @r_dmp:         damping ratio
%
%   @itPrams:       output structure

itPrams = struct('v1e_init',    v1e_init, ...
                 'gm_v1e_init', gm_v1e_init, ...
                 'K_it',        K_it, ...
                 'thd_brk',     thd_brk, ...
                 'r_dmp',       r_dmp);
end

