function verify_Hprime_contraction(num_trials)
fprintf('\n[3] Contraction: |H''(t)| < 1 for all t, with alpha,beta>0 and alpha+beta=2\n');
max_sup = 0;
for trial = 1:num_trials
    % sample ratio r = alpha/beta on a wide range
    r = 10^(rand()*4 - 2); % r in [10^-2, 10^2]
    beta  = 2/(1 + r);
    alpha = 2 - beta;
    t     = 20*(rand(1000,1)-0.5); % t in [-10,10]
    Hp    = (alpha*exp(t) - beta*exp(-t)) ./ (alpha*exp(t) + beta*exp(-t));
    max_sup = max(max_sup, max(abs(Hp)));
end
fprintf('  sup_t |H''(t)| over random samples = %.5f  %s\n', max_sup, passflag(max_sup < 1));
end

function s = passflag(tf)
if tf, s='[PASS]'; else, s='[WARN]'; end
end
