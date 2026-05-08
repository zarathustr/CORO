function verify_Gram_congruence(dims, num_trials, tol)
fprintf('\n[5] Gram congruence identity for one step\n');
for n = dims
    max_err = 0;
    for t = 1:num_trials
        R = sample_random_R(n, 1e3);
        if det(R) < 0, R(:,end) = -R(:,end); end
        G = R.'*R;
        rho = (n-1)/(n-2 + trace(G));

        % random admissible (alpha,beta) with alpha+beta=2
        r = 10^(rand()*4 - 2); beta = 2/(1+r); alpha = 2 - beta;

        C  = cofactor_matrix(R);
        Rp = rho*(alpha*R + beta*C);
        G1 = Rp.'*Rp;

        delta = gram_delta(G);
        T = (alpha*eye(n) + beta*delta*(G \ eye(n)));
        G2 = (rho^2) * (T * G * T);

        max_err = max(max_err, norm(G1 - G2, 'fro'));
    end
    max_err = max_err / n;
    fprintf('  n=%d: max ||G_{+}^{(direct)} - G_{+}^{(congruence)}||_F = %.2e  %s\n', ...
            n, max_err, passflag(max_err < 1e-7));
end
end

function s = passflag(tf)
if tf, s='[PASS]'; else, s='[WARN]'; end
end
