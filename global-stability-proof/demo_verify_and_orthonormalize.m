close all
clear all
clc

warning off
fprintf('=== Numerical verification & CORO orthonormalizer demo ===\n');

dims       = [3 4 5];     % test dimensions
num_trials = 100000;          % number of random trials per test
tol        = 1e-14;        % tolerance for identities
fprintf('\n[Section A] Verifying identities & inequalities used in the proof\n');

verify_cofactor_identities(dims, num_trials, tol);
verify_A2BplusC(dims, num_trials, tol);
verify_Hprime_contraction(num_trials);
verify_pz_monotonicity(dims, num_trials);
verify_Gram_congruence(dims, num_trials, tol);
verify_SO_invariance(dims, tol);

fprintf('\n[Section B] CORO orthonormalizer demo with equation (23) pre-scaling\n');

% Suppose the user provides a raw matrix B (not pre-scaled)
n  = 5;
B  = randn(n, n);

% Apply CORO pre-scaling per eq. (23): B/(n*max(abs(B(:))))
R0 = orthonorm(B);

opts.maxit = 20000;
opts.tol   = 1e-14;

% 1) Asymptotically optimal (alpha=beta=1)
out1 = orthonormalize_iter(B, 'balanced', opts);
R1 = out1.R;
hist1 = out1.hist;

% 2) Adaptive one-step–optimal (ratio from A,B,C)
out2 = orthonormalize_iter(B, 'adaptive', opts);
R2 = out2.R;
hist2 = out2.hist;

% 3) Dimension-only rule (+ optional switch to equal for polishing)
out3 = orthonormalize_iter(B, 'dimension', opts);
R3 = out3.R;
hist3 = out3.hist;

% Plot convergence: ||G_k - I||_F
figure; hold on; grid on;
semilogy(hist1.errG, 'o-','LineWidth',1.2); 
semilogy(hist2.errG, 's-','LineWidth',1.2); 
semilogy(hist3.errG, '^-','LineWidth',1.2);
xlabel('Iteration k'); ylabel('||R_k^T R_k - I||_F'); 
title('Convergence of CORO orthonormalizer (with eq. 23 pre-scaling)');
legend({'equal (\alpha=\beta=1)','adaptive (one-step optimal)','dimension-only (\beta=1/(n-2))'}, ...
       'Location','southwest');

% Quick summary
fprintf('\nFinal errors (||G-I||_F):\n');
fprintf('  equal:      %.3e (iters=%d)\n', hist1.errG(end), numel(hist1.errG));
fprintf('  adaptive:   %.3e (iters=%d)\n', hist2.errG(end), numel(hist2.errG));
fprintf('  dimension:  %.3e (iters=%d)\n', hist3.errG(end), numel(hist3.errG));

% Show final determinants
fprintf('Final dets:  equal=%.6f, adaptive=%.6f, dimension=%.6f\n', det(R1), det(R2), det(R3));
