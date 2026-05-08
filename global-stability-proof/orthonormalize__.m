function [R, hist] = orthonormalize(R0, mode, opts)
% Implements the iteration:
%   R_{k+1} = rho_k (alpha_k R_k + beta_k C(R_k)),
%   rho_k   = (n-1)/(n-2 + trace(R_k^T R_k)),
% where C(R) is the cofactor matrix (columns are the generalized cross products).
% We maintain alpha_k+beta_k=2 with alpha_k,beta_k>0.
%
% Equation (23) CORO pre-scaling (optional, default true):
%   R0 := R0 / ( n * max(abs(R0(:))) )
%
% mode:
%   'equal'     -> alpha=beta=1 (quadratic local rate)
%   'adaptive'  -> one-step–optimal alpha_k, beta_k
%   'dimension' -> beta=1/(n-2), alpha=2-beta  (n>=3)
%
% opts.tol         stopping tolerance on ||R^T R - I||_F
% opts.maxit       maximum number of iterations
% opts.apply_eq23  logical, apply equation (23) pre-scaling (default true)

if nargin < 3, opts = struct; end
if ~isfield(opts,'tol'),         opts.tol   = 1e-14; end
if ~isfield(opts,'maxit'),       opts.maxit = 2000;   end
if ~isfield(opts,'apply_eq23'),  opts.apply_eq23 = true; end

R = R0;
n = size(R,1);

hist.errG   = [];
hist.detR   = [];
hist.alpha  = [];
hist.beta   = [];
hist.rho    = [];
hist.eq23_s = [];

% Apply CORO pre-scaling per equation (23)
% if opts.apply_eq23
    [R, s] = coro_prescale(R);

hist.eq23_s = s;

for k = 1:opts.maxit
    if det(R) < 0, R(:,end) = -R(:,end); end  % enforce det >= 0 (orientation)
    G = R.'*R;
    errG = norm(G - eye(n),'fro');
    hist.errG(end+1) = errG; %#ok<AGROW>
    hist.detR(end+1) = det(R); %#ok<AGROW>

    if errG < opts.tol, break; end

    % choose parameters
    switch mode
        case 'equal'      % alpha=beta=1
            alpha = 1; beta = 1;
        case 'adaptive'   % one-step–optimal
            [alpha,beta] = choose_alpha_beta(G);
        case 'dimension'  % beta=1/(n-2), alpha=2-beta
            if n >= 3
                beta = 1/(n-2);
                alpha = 2 - beta;
                % Optional two-phase polishing: switch near the end
                if errG < 1e-1
                    alpha = 1; beta = 1;
                end
            else
                alpha = 1; beta = 1;  % safe fallback for n=2
            end
        otherwise
            error('Unknown mode: %s', mode);
    end

    hist.alpha(end+1) = alpha; %#ok<AGROW>
    hist.beta(end+1)  = beta;  %#ok<AGROW>

    % step size (normalizer)
    rho = (n-1)/(n-2 + trace(G));
    hist.rho(end+1) = rho; %#ok<AGROW>

    % update with cofactor matrix
    C  = cofactor_matrix(R);  % = det(R) * inv(R)^T
    R  = rho*(alpha*R + beta*C);
end
end
