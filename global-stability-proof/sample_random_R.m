function R = sample_random_R(n, kappa_max)
% Random square matrix with controlled condition number ~ kappa_max.
if nargin < 2, kappa_max = 1e3; end
[U,~] = qr(randn(n),0);
[V,~] = qr(randn(n),0);
smin  = 1/kappa_max; smax = kappa_max;
% random singular values spread across [smin, smax]
s = exp( linspace(log(smin), log(smax), n) ) .* (0.5 + rand(1,n));
R = U * diag(s) * V';
if det(R) < 0, R(:,end) = -R(:,end); end
end
