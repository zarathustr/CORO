function verify_pz_monotonicity(dims, num_trials)
fprintf('\n[4] Monotonicity toward z=1 of p(z)\n');
for n = dims
    ok = true;
    for trial = 1:num_trials
        r = 10^(rand()*4 - 2);
        beta  = 2/(1+r); alpha = 2 - beta;

        z_gt1 = 10.^rand(100,1);        % z in (1,10]
        z_lt1 = 10.^(-rand(100,1));     % z in [0.1,1)

        p_over_z_gt1 = ((n-1)./(n-2 + n*z_gt1.^2)) .* (alpha + beta./(z_gt1.^2));
        p_over_z_lt1 = ((n-1)./(n-2 + n*z_lt1.^2)) .* (alpha + beta./(z_lt1.^2));

        if any(p_over_z_gt1 >= 1) || any(p_over_z_lt1 <= 1)
            ok = false; break;
        end
    end
    fprintf('  n=%d: %s (random checks)\n', n, passflag(ok));
end
end

function s = passflag(tf)
if tf, s='[PASS]'; else, s='[WARN]'; end
end
