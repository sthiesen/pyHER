function [z_realiz, rec] = f_her4_HERs_NOnugget_rec(x_cal, y_cal, z_cal, x_target, y_target, her)
% Copy of functions/f_her4_HERs_NOnugget.m that records the random numbers (visiting order and
% Monte Carlo p-values) so the Python port can replay the same realisation. Algorithm unchanged.
    z_realiz = NaN(1,length(x_target));
    idx_ = randperm(length(x_target));
    rec.perm = idx_;
    rec.p_value = NaN(1,length(x_target));
    x_cal_ = x_cal; y_cal_ = y_cal; z_cal_ = z_cal;
    for i = 1:length(idx_)
        euc_distance_ = NaN(1,length(x_cal));
        for j = 1:length(x_cal)
            euc_distance_(j) = f_euclidean_dist(x_cal(j), y_cal(j), x_target(idx_(i)), y_target(idx_(i)));
        end
        [dist_, idx_cal_] = min(euc_distance_);
        if dist_ <= 10e-6
            z_realiz(idx_(i)) = z_cal(idx_cal_);
        else
            [pmf, ~] = f_her3_predict_NOnugget(x_cal_, y_cal_, z_cal_, x_target(idx_(i)), y_target(idx_(i)), her);
            p_value = rand;
            rec.p_value(idx_(i)) = p_value;
            ccdf = [0 cumsum(cell2mat(pmf))];
            k = find(ccdf >= p_value, 1);
            z_realiz(idx_(i)) = interp1(ccdf([k-1 k]), her.edges_z([k-1 k]), p_value);
            x_cal_ = [x_cal_; x_target(idx_(i))]; y_cal_ = [y_cal_; y_target(idx_(i))]; z_cal_ = [z_cal_; z_realiz(idx_(i))];
        end
    end
end
