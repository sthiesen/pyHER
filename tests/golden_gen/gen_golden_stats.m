%% Golden outputs for PMF statistics and performance metrics, from the ORIGINAL MATLAB functions.
% Inputs: the golden HER3 PMFs (25x25 grid) and the Jura validation set; every validation point
% takes the PMF of its nearest grid node. The metric code mirrors the block in
% sHER_E_0304_Extract_pmf_statistics_performance_and_plot.m (grid indexing replaced by this lookup).
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..', '..', 'HERs');
gold = fullfile(here, '..', 'golden');
addpath(fullfile(root, 'functions'));
addpath(fullfile(here, 'shims'), '-end');   % corr / nanmean only if the toolbox is missing
inp = load(fullfile(gold, 'inputs.mat'));
h1 = load(fullfile(gold, 'her1.mat')); her = h1.her1;
g = load(fullfile(gold, 'her3_pred.mat'));
her.z_thresh = round(log10(50), 5);

z_cal = round(inp.z_cal, 5); z_val = round(inp.z_val, 5);
x_val = inp.x_val; y_val = inp.y_val;
P = g.pmf_pred_mat;                       % [T x n_bins]
cellP = mat2cell(P, ones(size(P,1),1), size(P,2))';   % {1,T}

% --- PMF statistics of the whole grid
[z_entropy, z_mean, z_mode, z_prob] = f_extract_pmf_statistics(cellP, her.edges_z, her.bin_centers_edges_z, her.z_thresh);
z_prob_nothresh = nan; [~,~,~,zp_none] = f_extract_pmf_statistics(cellP(1:3), her.edges_z, her.bin_centers_edges_z, NaN);

% --- validation points -> nearest grid node
gx = g.gx(:); gy = g.gy(:);
nn = zeros(numel(z_val),1);
for i = 1:numel(z_val)
    [~, nn(i)] = min(hypot(gx - x_val(i), gy - y_val(i)));
end
pmf_val = P(nn,:);
PMF_pred_ = mat2cell(pmf_val, ones(numel(nn),1), size(P,2));   % {V,1}, as in the script
z_pred_ = round(z_mean(nn), 5)';
prob_thres_pred_ = round(z_prob(nn), 5)';
z_true_ = z_val;
perf.marginal_prob_thres_cal = sum(double(z_cal) > her.z_thresh) / length(z_cal);
[perf.error_sign, perf.RMSE, perf.ME, perf.MAE, perf.NSE, perf.correl_true_pred, perf.correl_true_residue] = f_performance_det(z_pred_, z_true_);
perf.DKL_score_mean = f_performance_prob(z_true_', PMF_pred_', ones(numel(PMF_pred_),1)', her.edges_z);

obs_larger_thres_ = double(z_true_ > her.z_thresh);
obs_smaller_thres_ = double(z_true_ <= her.z_thresh);
perf.DKL_score_01_largersmaller_thres = zeros(numel(PMF_pred_),1);
for target_ = 1 : numel(PMF_pred_)
    if obs_smaller_thres_(target_) == 1
        perf.DKL_score_01_largersmaller_thres(target_,1) = (log2(obs_smaller_thres_(target_)) - log2(1-prob_thres_pred_(target_)))*obs_smaller_thres_(target_);
    else
        perf.DKL_score_01_largersmaller_thres(target_,1) = (log2(obs_larger_thres_(target_)) - log2(prob_thres_pred_(target_)))*obs_larger_thres_(target_);
    end
end
perf.DKL_score_01_largersmaller_thres_mean = mean(perf.DKL_score_01_largersmaller_thres);
perf.DKL_score_1_larger_thres = zeros(numel(PMF_pred_),1);
for target_ = 1 : numel(PMF_pred_)
    if obs_larger_thres_(target_) == 0
        perf.DKL_score_1_larger_thres(target_,1) = NaN;
    else
        perf.DKL_score_1_larger_thres(target_,1) = (log2(obs_larger_thres_(target_)) - log2(prob_thres_pred_(target_)))*obs_larger_thres_(target_);
    end
end
perf.DKL_score_1_larger_thres_mean = nanmean(perf.DKL_score_1_larger_thres);

perf.prob_thres = round([0:0.01:1]',2);
perf.index_marg = find(perf.prob_thres == round(perf.marginal_prob_thres_cal,2));
for i = 1:length(perf.prob_thres)
    contamin_pred_ = double(prob_thres_pred_ > perf.prob_thres(i));
    [perf.TP(i), perf.TN(i), perf.FP(i), perf.FN(i), perf.TPR(i), perf.TNR(i), perf.FPR(i), perf.Accuracy(i), perf.F1_score(i), perf.MCC(i), perf.DistanceROC(i)] = f_ROC(obs_larger_thres_, contamin_pred_);
    perf.misclassication(i,1) = 100 * (1 - perf.Accuracy(i));
    perf.contamination(i,1) = 100 * (sum(contamin_pred_) ./ numel(PMF_pred_));
end
[perf.G, perf.ksi_fraction, perf.edges_interv, perf.interv_size] = f_G(PMF_pred_, her.edges_z, z_true_);
perf.ksi_fraction_uq_lq = perf.ksi_fraction(find(perf.edges_interv(:,1)==0.25));
perf.misclass_mean = 100*(sum(double(z_val > her.z_thresh) ~= double(z_pred_ > her.z_thresh)))/numel(PMF_pred_);
perf.misclass_min = min(perf.misclassication(:));
perf.misclass_marginal = perf.misclassication(perf.index_marg);
perf.Accuracy_marginal = 100*perf.Accuracy(perf.index_marg);
perf.F1_score_marginal = perf.F1_score(perf.index_marg);
perf.MCC_marginal = perf.MCC(perf.index_marg);
perf.DistanceROC_marginal = perf.DistanceROC(perf.index_marg);

% the non-HER helper f_perf must agree with the script block
perf_f = f_perf(z_true_, z_pred_, prob_thres_pred_, her.z_thresh, perf.marginal_prob_thres_cal, perf.prob_thres, perf.index_marg);

save(fullfile(gold, 'stats.mat'), 'z_entropy','z_mean','z_mode','z_prob','zp_none','nn','pmf_val', ...
     'z_val','z_cal','perf','perf_f','-v7');
disp('STATS DONE'); exit(0)
