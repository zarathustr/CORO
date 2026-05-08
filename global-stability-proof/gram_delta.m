function delta = gram_delta(G)
% Robustly compute delta = sqrt(det(G)) for SPD/PD G
% via singular values/log to avoid under/overflow.
s = svd(G);
delta = exp(0.5*sum(log(max(s, eps))));
end
