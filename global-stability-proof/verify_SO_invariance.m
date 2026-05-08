function verify_SO_invariance(dims, tol)
fprintf('\n[6] Invariance: if R in SO(n) then one step returns R\n');
for n = dims
    % random orthogonal Q with det=1
    [Q,~] = qr(randn(n));
    if det(Q) < 0, Q(:,1) = -Q(:,1); end
    R = Q;
    G = R.'*R;
    rho = (n-1)/(n-2 + trace(G));     % = 1/2
    alpha = 1; beta = 1;              % any alpha+beta=2 works

    C  = cofactor_matrix(R);          % = R for det=1 orthogonal
    Rp = rho*(alpha*R + beta*C);

    err = norm(Rp - R, 'fro');
    fprintf('  n=%d: ||R_{+} - R||_F = %.2e  %s\n', n, err, passflag(err < 1e2*tol));
end
end

function s = passflag(tf)
if tf, s='[PASS]'; else, s='[WARN]'; end
end
