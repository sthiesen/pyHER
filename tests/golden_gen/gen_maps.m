%% MATLAB maps from the ORIGINAL functions with the golden (MATLAB-optimised) weights and classes.
% Output goes to the folder given in OUTDIR (kept out of the repository: the PMF matrix is large).
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..', '..', 'HERs');
gold = fullfile(here, '..', 'golden');
addpath(fullfile(root, 'functions'));
addpath(here); addpath(fullfile(here, 'shims'), '-end');
inp = load(fullfile(gold, 'inputs.mat'));
r2 = load(fullfile(gold, 'her2.mat')); her = r2.her2;      % infogram classes + AND/OR weights + alpha/beta
x_cal = inp.x_cal; y_cal = inp.y_cal; z_cal = round(inp.z_cal, 5);

% --- HER3: full 0.05 grid (121 x 121)
[gx, gy] = meshgrid(0:0.05:6, 0:0.05:6);
gx = round(gx,2); gy = round(gy,2);
tic; [pmf_cell, idx_zero] = f_her3_predict(x_cal, y_cal, z_cal, gx(:), gy(:), her); fprintf('HER3 full grid %.1fs\n', toc);
pmf = cell2mat(pmf_cell(:));
[H, zmean, zmode, zprob] = f_extract_pmf_statistics(pmf_cell, her.edges_z, her.bin_centers_edges_z, her.z_thresh);
save(fullfile(OUTDIR, 'maps_matlab.mat'), 'gx','gy','pmf','idx_zero','H','zmean','zmode','zprob','-v7');

% --- HERs4: one realisation on a 0.1 grid (61 x 61), recording the random numbers
[sx, sy] = meshgrid(0:0.1:6, 0:0.1:6);
sx = round(sx,2); sy = round(sy,2);
rng(7);
tic; [z_realiz, rec] = f_her4_HERs_rec(x_cal, y_cal, z_cal, sx(:), sy(:), her); fprintf('HERs4 61x61 %.1fs\n', toc);
save(fullfile(OUTDIR, 'sim_matlab.mat'), 'sx','sy','z_realiz','rec','-v7');
disp('MAPS DONE'); exit(0)
