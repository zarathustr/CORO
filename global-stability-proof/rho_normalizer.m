function rho = rho_normalizer(R)
% rho = (n-1) / (n-2 + sum_j ||r_j||^2) = (n-1)/(n-2 + trace(R'*R))
    n = size(R,1);
    rho = (n-1) / ( (n-2) + trace(R.'*R) );
end
