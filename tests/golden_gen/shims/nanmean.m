function m = nanmean(x)
% Minimal stand-in for the Statistics Toolbox nanmean() (vector input).
m = mean(x(~isnan(x)));
end
