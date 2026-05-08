function out = orthonormalize_iter(R0, policy, opts)
% policy: 'adaptive' (one-step optimal), 'balanced' (alpha=beta=1),
%         'dimonly' (beta=1/(n-2), alpha=2-beta)
% opts: struct with fields: maxit, tol, verbose, switch_tol (for 2-phase)

    if nargin < 3, opts = struct; end
    if ~isfield(opts,'maxit'),      opts.maxit = 20000; end
    if ~isfield(opts,'tol'),        opts.tol = 1e-14; end
    if ~isfield(opts,'verbose'),    opts.verbose = true; end
    if ~isfield(opts,'switch_tol'), opts.switch_tol = 1e-2; end

    n = size(R0,1);
    B = R0;
    mx = n * max(abs(B(:)));
    if mx > 0
        B = B / mx;
    end
    R = B;
    hist.normGI = zeros(opts.maxit,1);
    hist.det    = zeros(opts.maxit,1);
    hist.alpha  = zeros(opts.maxit,1);
    hist.beta   = zeros(opts.maxit,1);

    phase2 = false;

    for k = 1:opts.maxit
        G = R.'*R;
        hist.errG(k) = norm(G - eye(n), 'fro');
        hist.det(k)    = det(R);

        switch policy
            case 'adaptive'
                [alpha,beta] = alpha_beta_one_step_opt(R);
            case 'balanced'
                alpha = 1; beta = 1;
            case 'dimension'
                beta = 1/(n-2); alpha = 2 - beta;
                % optional two-phase switch
                if ~phase2 && hist.normGI(k) < opts.switch_tol
                    alpha = 1; beta = 1; phase2 = true;
                end
            otherwise
                error('Unknown policy.');
        end

        hist.alpha(k) = alpha;
        hist.beta(k)  = beta;

        Rnext = orthonormalize_step(R, alpha, beta);
        if norm(Rnext.'*Rnext - eye(n), 'fro') < opts.tol
            R = Rnext;
            hist.normGI(k+1:end) = [];
            hist.det(k+1:end) = [];
            hist.alpha(k+1:end) = [];
            hist.beta(k+1:end)  = [];
            break;
        end
        R = Rnext;
    end

    out.R = R;
    out.hist = hist;
end
