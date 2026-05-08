function verify_A2BplusC(dims, num_trials, tol)
fprintf('\n[2] Nonnegativity: A-2B+C = sum_i (sqrt(lambda_i) - delta/sqrt(lambda_i))^2 >= 0\n');
for n = dims
    min_margin = inf; max_diff = 0;
    for t = 1:num_trials
        R = sample_random_R(n, 1);
        G = R.'*R;
        A = trace(G);
        delta = gram_delta(G);
        B = n*delta;
        C = (delta^2)*trace(G \ eye(n));
        lhs = A - 2*B + C;                            % should be >= 0
        lam = eig(G);
        sqsum = sum( (sqrt(lam) - delta./sqrt(lam)).^2 );
        min_margin = min(min_margin, lhs);
        max_diff   = max(max_diff, abs(lhs - sqsum));
    end
    fprintf('  n=%d: min(A-2B+C)=%.3e,  max|lhs - spectral|=%.2e  %s\n', ...
        n, min_margin, max_diff, passflag(min_margin >= -1e2*tol && max_diff < 1e2*tol));
end
end

function s = passflag(tf)
if tf, s='[PASS]'; else, s='[WARN]'; end
end
