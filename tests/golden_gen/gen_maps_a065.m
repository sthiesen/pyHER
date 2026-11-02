%% MATLAB maps from the ORIGINAL functions with the golden (MATLAB-optimised) weights and classes.
% Output goes to the folder given in OUTDIR (kept out of the repository: the PMF matrix is large).
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..', '..', 'HERs');
gold = fullfile(here, '..', 'golden');
addpath(fullfile(root, 'functions'));
addpath(here); addpath(fullfile(here, 'shims'), '-end');
inp = load(fullfile(gold, 'inputs.mat'));
r2 = load(fullfile(gold, 'her2.mat')); her = r2.her2; her.best_alpha = 0.65; her.best_beta = 0;  % paper values
x_cal = inp.x_cal; y_cal = inp.y_cal; z_cal = round(inp.z_cal, 5);

% --- HER3: full 0.05 grid (121 x 121)
[gx, gy] = meshgrid(0:0.05:6, 0:0.05:6);
gx = round(gx,2); gy = round(gy,2);
tic; [pmf_cell, idx_zero] = f_her3_predict(x_cal, y_cal, z_cal, gx(:), gy(:), her); fprintf('HER3 full grid %.1fs\n', toc);
pmf = cell2mat(pmf_cell(:));
[H, zmean, zmode, zprob] = f_extract_pmf_statistics(pmf_cell, her.edges_z, her.bin_centers_edges_z, her.z_thresh);
save(fullfile(OUTDIR, 'maps_matlab_a065.mat'), 'gx','gy','pmf','idx_zero','H','zmean','zmode','zprob','-v7');

disp('MAPS DONE'); exit(0)
