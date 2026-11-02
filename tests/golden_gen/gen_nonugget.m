%% NOnugget (paper) version: golden outputs from the ORIGINAL f_her3_predict_NOnugget / f_her4_HERs_NOnugget.
% Weights = the golden fmincon weights; slopes recomputed exactly as f_her2_weight_thres_NOnugget does;
% alpha = 0.65, beta = 0, n_neighbor = 30 (the paper's settings). OUTDIR: folder for the large files.
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..', '..', 'HERs');
gold = fullfile(here, '..', 'golden');
addpath(fullfile(root, 'functions')); addpath(here); addpath(fullfile(here, 'shims'), '-end');
inp = load(fullfile(gold, 'inputs.mat'));
r2 = load(fullfile(gold, 'her2.mat')); her = r2.her2;
x_cal = inp.x_cal; y_cal = inp.y_cal; z_cal = round(inp.z_cal, 5);
her.best_alpha = 0.65; her.best_beta = 0; her.n_neighbor = 30;
% slopes as in f_her2_weight_thres_NOnugget.m
her.w_OR_slope = NaN(1,length(her.best_w_OR));
for i = 2:length(her.best_w_OR)
    her.w_OR_slope(i) = (her.best_w_OR(i) - her.best_w_OR(i-1)) / (her.edges_distance_classes_range(i) - her.edges_distance_classes_range(i-1));
end
her.w_AND_slope = NaN(1,length(her.best_w_AND));
for i = 2:length(her.best_w_AND)
    her.w_AND_slope(i) = (her.best_w_AND(i) - her.best_w_AND(i-1)) / (her.edges_distance_classes_range(i) - her.edges_distance_classes_range(i-1));
end
w_OR_slope = her.w_OR_slope; w_AND_slope = her.w_AND_slope;

% ---- small, permanent golden set: 25x25 grid + 5 targets exactly on data points
[gx, gy] = meshgrid(0:0.25:6, 0:0.25:6); gx = round(gx,2); gy = round(gy,2);
tx = [gx(:); x_cal(1:5)]; ty = [gy(:); y_cal(1:5)];
[pc, idx_zero] = f_her3_predict_NOnugget(x_cal, y_cal, z_cal, tx, ty, her);
pmf_mat = cell2mat(pc(:));
[sx, sy] = meshgrid(0:0.5:6, 0:0.5:6); sx = round(sx,2); sy = round(sy,2);
sx = [sx(:); x_cal(1:3)]; sy = [sy(:); y_cal(1:3)];
rng(11); [z_realiz, rec] = f_her4_HERs_NOnugget_rec(x_cal, y_cal, z_cal, sx, sy, her);
rng(11); z_orig = f_her4_HERs_NOnugget(x_cal, y_cal, z_cal, sx, sy, her);
assert(isequaln(z_orig, z_realiz), 'recording copy diverges from the original');
save(fullfile(gold, 'nonugget.mat'), 'w_OR_slope','w_AND_slope','tx','ty','pmf_mat','idx_zero','sx','sy','z_realiz','rec','-v7');

% ---- full grid (large, not stored in the repository)
[gx, gy] = meshgrid(0:0.05:6, 0:0.05:6); gx = round(gx,2); gy = round(gy,2);
tic; [pc, idx_zero] = f_her3_predict_NOnugget(x_cal, y_cal, z_cal, gx(:), gy(:), her); fprintf('HER3 NOnugget full grid %.1fs\n', toc);
pmf = cell2mat(pc(:));
[H, zmean, zmode, zprob] = f_extract_pmf_statistics(pc, her.edges_z, her.bin_centers_edges_z, her.z_thresh);
save(fullfile(OUTDIR, 'maps_matlab_nonugget.mat'), 'gx','gy','pmf','idx_zero','H','zmean','zmode','zprob','-v7');
disp('NONUGGET DONE'); exit(0)
