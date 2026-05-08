function [alpha,beta] = choose_alpha_beta(G)
% One-step–optimal parameters subject to alpha+beta=2:
%   A = tr(G),  delta = sqrt(det G),  B = n*delta,  C = delta^2 tr(G^{-1})
%   beta* = 2(A-B)/(A-2B+C), clipped to (0,2); alpha*=2-beta*.

n     = size(G,1);
A     = trace(G);
delta = gram_delta(G);
B     = n*delta;
C     = (delta^2)*trace(G \ eye(n));  % trace(G^{-1})
denom = A - 2*B + C;

if denom <= 1e-14
    beta = 1;                 % isotropic (or nearly) -> equal weights
else
    beta = 2*(A - B)/denom;
end
beta  = min(max(beta,1e-6), 2 - 1e-6);
alpha = 2 - beta;
end
