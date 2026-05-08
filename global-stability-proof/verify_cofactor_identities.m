function verify_cofactor_identities(dims, num_trials, tol)
fprintf('\n[1] Cofactor identities: R^T C(R) = det(R) I and C(R)^T C(R) = det(R)^2 (R^T R)^{-1}\n');
for n = dims
    max_err1 = 0; max_err2 = 0;
    for t = 1:num_trials
        R = sample_random_R(n, 1e3);
        G = R.'*R; 
        C = cofactor_matrix(R); 
        d = det(R);
        E1 = norm(R.'*C - d*eye(n), 'fro');
        E2 = norm((C.'*C - (d^2)*(G \ eye(n))) / trace(C.'*C)^2, 'fro');
        max_err1 = max(max_err1, E1);
        max_err2 = max(max_err2, E2);
    end
    fprintf('  n=%d: max ||R^T C - det(R)I||_F = %.2e,  max ||C^T C - det(R)^2 G^{-1}||_F = %.2e  %s\n', ...
        n, max_err1, max_err2, passflag(max([max_err1,max_err2]) < 5e-7));
end
end

function s = passflag(tf)
if tf, s='[PASS]'; else, s='[WARN]'; end
end
