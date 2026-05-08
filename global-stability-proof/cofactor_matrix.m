function C = cofactor_matrix(R)
% Returns the cofactor matrix C(R) (classical adjugate).
% For nonsingular R: C(R) = det(R) * inv(R)^T.
C = det(R) * (R \ eye(size(R)))';
end
