function r = corr(a, b)
% Minimal stand-in for the Statistics Toolbox corr() (Pearson, two vectors); only used to run the
% original f_performance_det.m on a machine without that toolbox.
c = corrcoef(a(:), b(:));
r = c(1,2);
end
