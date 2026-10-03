# A concrete finite-reference detection boundary

Analytical consequence of the frozen design, no algorithm change or extra
simulation. This is an elementary cap/self-consistency argument, not claimed
as an original general lower-bound theorem.

On a superuniform grid with first support point t1, the normalized decreasing
calibrator e satisfies t1*e(t1)<=sum_k(tk-t(k-1))*e(tk)<=1. Thus e<=1/t1.
For n=M+1 predictive reference ranks, component caps are n (projection) and
n/3 (ordinary). Their convex mixture is bounded by n. If e-BH rejects R>0
hypotheses among m, at least the R-th evidence must be at least m/(qR), so

    R >= ceil(m/(q*n)) =: k_min.

For the frozen m512,q.05,n4096, k_min=3 (pure ordinary:8). PILOT weighting
can further reduce the actual cap, not increase it. These are not merely
numerical underflow artifacts. Exact calibration itself has a resolution cost.

Suppose there are s true signed alternatives, 0<s<k_min. Every nonempty
rejection set has FDP>=1-s/k_min. Under the model FDR guarantee,

    Pr(R>0) <= q/(1-s/k_min),
    Power=E[TP/s] <= Pr(R>0) <= q/(1-s/k_min).

For s=1 this gives Power<=7.5% (ordinary cap: <=5.7143%). This concerns
extremely sparse alternatives; it does not bound the current s51-ish scenario
by these numbers. The bound is conservative and not an attainability result.
No new experiment is necessary to establish this algebraic implication.

Avoiding a minimum-rejection-count restriction greater than one would require
n>=m/q (for projection cap alone), or n>=3m/q for pure ordinary. With m512,
that is n>=10240 or30720 (first allowed dyadic interface sizes16384/32768), before accounting for other calibration/learning
losses. This is a NECESSARY reference-resolution condition for this algorithm,
not a sufficient guarantee of detection, not a universal lower bound for all
tests, and not an instruction to enlarge C001 or begin another optimization.

Separately, matrix-pivot ratios A/kappa and their median/error distribution do
not depend on rho within the matched compound model. Relative calibration
uncertainty therefore does not itself diverge as rho approaches1. A larger
absolute kappa, numerical conditioning and genuine target/calibration mismatch
are different issues. No general sample-complexity or minimax rate is claimed.
