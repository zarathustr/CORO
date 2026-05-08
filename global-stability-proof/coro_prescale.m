function [Bscaled, s] = coro_prescale(B)
% CORO pre-scaling according to equation (23):
%   B_scaled = B / ( n * max(abs(B(:))) )
% where n is the matrix dimension (square case here, n>=3).
% If max(abs(B(:))) == 0 (zero matrix), no scaling is applied and s=1.
%
% Returns:
%   Bscaled : scaled matrix
%   s       : scale factor used (s = n*max(abs(B(:))))

[n,m] = size(B);
assert(n==m, 'This CORO implementation assumes square matrices.');
% M = max(abs(B(:)));
M = (n + norm(B, 'fro').^2) ./ (2 * n);
if M == 0
    s = 1;
    Bscaled = B;
else
    s = n * M;
    Bscaled = B / s;
end
end
