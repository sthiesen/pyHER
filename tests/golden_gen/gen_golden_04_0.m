%% Generate golden reference outputs from the ORIGINAL MATLAB code (sHER_E_04_0 settings).
% Original code in HERs/ is not modified. Outputs go to ../golden/*.mat (v7, scipy-readable).
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..', '..', 'HERs');
outdir = fullfile(here, '..', 'golden');
addpath(fullfile(root, 'functions'));
addpath(here);
load(fullfile(root, '1.0.raw data', 'Jura_dataset_Pb_log10.mat'));

% ---- settings copied from sHER_E_04_0_Jura_struct.m ----
her = struct();
her.txt = txt; her.dim_cal = dim_cal;
her.lag_dist = 0.07; her.binwidth = 0.015; her.binwidth_z = 0.015;
her.n_neighbor = 9; her.z_thresh = log10(50);
her.aggregation_type = 'andor';
z = round(z,5); z_cal = round(z_cal,5); z_val = round(z_val,5);
her.z_thresh = round(her.z_thresh,5);
fprintf('dim_cal=%d dim_val=%d\n', dim_cal, dim_val);

% ---- inputs ----
save(fullfile(outdir,'inputs.mat'), 'x_cal','y_cal','z_cal','x_val','y_val','z_val', ...
     'idx_cal','idx_val','x','y','z','txt','dim_cal','dim_val','-v7');

% ---- HER1 ----
tic; her1 = f_her1_infogram(x_cal, y_cal, z_cal, her); fprintf('HER1 %.1fs\n', toc);
her_in = her;
save(fullfile(outdir,'her1.mat'), 'her_in','her1','-v7');

% ---- HER2 (threshold objective) ----
tic; her2 = f_her2_weight_thres(x_cal, y_cal, z_cal, her1); fprintf('HER2 %.1fs\n', toc);
save(fullfile(outdir,'her2.mat'), 'her2','-v7');

% ---- HER3 prediction on a coarse grid (full 0.05 grid is 14641 targets) ----
[gx, gy] = meshgrid(0:0.25:6, 0:0.25:6);
gx = round(gx,2); gy = round(gy,2);
tic; [pmf_pred, idx_zero] = f_her3_predict(x_cal, y_cal, z_cal, gx(:), gy(:), her2); fprintf('HER3 %.1fs\n', toc);
pmf_pred_mat = cell2mat(pmf_pred(:));      % [T x n_bins_z]
save(fullfile(outdir,'her3_pred.mat'), 'gx','gy','pmf_pred_mat','idx_zero','-v7');

% ---- HERs4: one realisation on a coarser grid, with recorded random numbers ----
[sx, sy] = meshgrid(0:0.5:6, 0:0.5:6);
sx = round(sx,2); sy = round(sy,2);
rng(1);
tic; [z_realiz, rec] = f_her4_HERs_rec(x_cal, y_cal, z_cal, sx(:), sy(:), her2); fprintf('HERs4 %.1fs\n', toc);
% sanity: the recording copy must equal the original function under the same seed
rng(1); z_orig = f_her4_HERs(x_cal, y_cal, z_cal, sx(:), sy(:), her2);
assert(isequaln(z_orig, z_realiz), 'recording copy diverges from original f_her4_HERs');
save(fullfile(outdir,'hers4.mat'), 'sx','sy','z_realiz','rec','-v7');

% (leaf-function references are produced by gen_golden_leaf.m)
disp('GOLDEN DONE');
