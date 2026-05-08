function C = cofactor_cols(R)
% C = cofactor_cols(R)
% Returns the matrix C(R) whose j-th column is s_j r_j^{\otimes}.
% Numerically: C(R) = adj(R)^T. For nonsingular R: adj(R) = det(R)*inv(R).
% For near-singular R we fall back to minors (O(n^4), but robust for small n).

    n = size(R,1);
    % Try fast path first
    rc = rcond(R);
    if rc > 1e-12 && isfinite(det(R))
        C = det(R) * inv(R)';   % adj(R)^T
        return;
    end

    % Fallback: explicit cofactors (works also for singular R)
    C = zeros(n);
    for j = 1:n
        for i = 1:n
            M = R; M(i,:) = []; M(:,j) = [];
            C(i,j) = (-1)^(i+j) * det(M);   % this is the cofactor matrix = adj(R)^T
        end
    end
end
