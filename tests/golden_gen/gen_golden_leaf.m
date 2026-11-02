%% Golden outputs for the leaf functions (no toolbox needed).
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..', '..', '..', 'HERs');
outdir = fullfile(here, '..', 'golden');
addpath(fullfile(root, 'functions'));
rng(2);
p = rand(5,8); p = p ./ sum(p,2); w = rand(5,1); w = w ./ sum(w);
leaf.pmfs = p; leaf.w = w;
leaf.entropy = f_entropy(p(1,:));
leaf.linear = f_linear_aggregation(p, w);
leaf.loglinear = f_loglinear_aggregation(p, w);
rng(3); leaf.mc_p = rand; rng(3);
leaf.montecarlo = f_montecarlosim(p(1,:), 0:8);
leaf.edges = 0:8;
% threshold performance score: z_true, 3 two-bin PMFs, edges [0 1.5 3]
leaf.perf_z = [0.5 2.0 1.0];
leaf.perf_pmf = [0.7 0.3; 0.2 0.8; 0.55 0.45];
leaf.perf_edges = [0 1.5 3];
pm = mat2cell(leaf.perf_pmf, ones(3,1), 2)';
leaf.perf = f_performance_prob(leaf.perf_z, pm, ones(1,3), leaf.perf_edges);
save(fullfile(outdir,'leaf.mat'), 'leaf','-v7');
disp('LEAF DONE'); exit(0)
