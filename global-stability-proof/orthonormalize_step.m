function Rnext = orthonormalize_step(R, alpha, beta)
% Implements: R_{+} = rho * (alpha*R + beta * C(R)),  with rho from (24)
    C = cofactor_cols(R);
    rho = rho_normalizer(R);
    Rnext = rho * (alpha*R + beta*C);
end
