function [alpha, beta, ratio] = alpha_beta_one_step_opt(R)
% One-step optimal parameters (minimize tr(G_next) with alpha+beta=2).
% A = tr(G), B = n*delta, C = delta^2 tr(G^{-1}), delta = sqrt(det G).
% Returns alpha,beta in [0,2] (clipped) and ratio = alpha/beta.

    G = R.'*R;
    n = size(R,1);
    % Ensure positive definite G numerically
    [V,D] = eig((G+G')/2); lam = max(real(diag(D)), 1e-18);
    G = V*diag(lam)*V';
    delta = sqrt(prod(lam));
    A = sum(lam);
    B = n*delta;
    C = delta^2 * sum(1./lam);

    denom = (A - 2*B + C);
    if abs(denom) < 1e-14
        beta = 1; alpha = 1; ratio = 1;
        return;
    end
    beta  = 2*(A - B) / denom;
    beta  = min(max(beta, 1e-3), 2-1e-3);  % robust clipping
    alpha = 2 - beta;
    ratio = alpha / beta;
end
