%% Golden outputs for the 03 (single-bin DKL) optimisation from the ORIGINAL functions.
% VARIANT = '03_0' : sHER_E_03_0 settings, f_her2_weight            (nugget, n_neighbor = 30)
% VARIANT = '03_1' : sHER_E_03_1 settings, f_her2_weight_NOnugget   (no nugget, n_neighbor = 9)
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..', '..', 'HERs');
gold = fullfile(here, '..', 'golden');
addpath(fullfile(root, 'functions')); addpath(here);
inp = load(fullfile(gold, 'inputs.mat'));
x_cal = inp.x_cal; y_cal = inp.y_cal; z_cal = round(inp.z_cal, 5);

her = struct();
her.txt = inp.txt; her.dim_cal = inp.dim_cal;
her.lag_dist = 0.07; her.binwidth = 0.015; her.binwidth_z = 0.015;
her.z_thresh = round(log10(50), 5);
her.aggregation_type = 'andor';
if strcmp(VARIANT, '03_0'), her.n_neighbor = 30; else, her.n_neighbor = 9; end
her = f_her1_infogram(x_cal, y_cal, z_cal, her);
tic;
if strcmp(VARIANT, '03_0')
    her2 = f_her2_weight(x_cal, y_cal, z_cal, her);
else
    her2 = f_her2_weight_NOnugget(x_cal, y_cal, z_cal, her);
end
fprintf('HER2 %s %.1fs\n', VARIANT, toc);
save(fullfile(gold, ['her2_' VARIANT '.mat']), 'her2', '-v7');
disp(['DONE ' VARIANT]); exit(0)
