# Conditional adaptation versus moment cost — proposition under review

2026-09-18 18:32HKT. This concerns A0.7's selected target reference, not all
possible safe borrowing algorithms. No Power theorem or G1 acceptance claimed.
It supplies a concrete quantitative target for a later joint-budget proof.

## Definitions

Condition on W and independent meta. Let Y=Et be target centered error, with
density f(y) proportional to product_i g0(y+b_i), where
g0(x)=8exp(2x)/(1+2exp(x))^3 and b_i are fixed shifts including meta centering.
Es independent of Y. Let g be a nonincreasing [0,1] borrowing probability,
u=Es+s-Y, legal drift slack s>=0, and

    h_s(y)=E_sour[1-g(Es+s-y)], piT(s)=E[h_s(Y)].

Use independent randomizer implementing this gate. piT is nondecreasing in s;
h_s is nonincreasing in y. Assume 0<piT(s), and positive finite moments used below.
No actual unknown slack enters a deployed selector.

## P-MOM: an exact, finite conditional moment penalty

For 0<r<2Nt put d_N(r)=4*atanh(r/(3Nt)). Then

    E[exp(-rY) | targetbranch,s] / E[exp(-rY)]
       >= piT(s+d_N(r))/piT(s) >=1.                     (1)

Proof. Set V=-log f. Differentiating the logF product gives

    V'(y+d)-V'(y)
      =3 sum_i [expit(y+b_i+log2+d)-expit(y+b_i+log2)]
      <=3Nt*tanh(d/4).

The logistic increment reaches its maximum tanh(d/4) at its centered interval.
For d=d_N(r), the tilted density f_r(y)=exp(-r y)f(y)/M(-r) obeys

    d/dy log[f_r(y)/f(y+d)]
       =-r-V'(y)+V'(y+d) <=0.

Both densities integrate to1. The decreasing likelihood ratio implies
Y_r <=_st Y-d. Therefore E_r h_s(Y)>=E h_s(Y-d)=piT(s+d).
Finally the left side of(1) is E_r h_s(Y)/piT(s), by the exact selected density.
The moment exists because the left density tail is exp(2Nt*y) times a constant
for every fixed finiteW; no uniform moment bound overW is being claimed.

A coarser general curvature version replaces d_N(r) by r/A whenever V''<=A;
here A=3Nt/4. The logistic increment gives a slightly stronger finite shift.
d_N is the largest translation with this universal decreasing-LR certificate
for all residual patterns: at W=0 the summed increment attains3Nt*tanh(d/4).
This is NOT a claim of the largest stochastic-order translation or sharp Power
bound; the actual stochastic shift or model-specific penalty may be larger.

## Actual learned-shape pivot moments, without assuming t20

In whitened coordinates write V=Z/sqrt(H11 tr(Y' H^-1 Y)/20), with Z~N(0,1),
Y a4x5 independent standard-normal matrix, H SPD independent of (Z,Y). This is
the equivalent representation of the actual H/eigen/U reference, not a new
shape algorithm. For each column y_j, Cauchy gives

    (e1'y_j)^2 <= H11*y_j'H^-1*y_j.

Thus on this coupling V_+ <= sqrt20*Z_+/||e1'Y|| = 2*(t5)_+.
For every 0<alpha<5, 0<E[V_+^alpha]<infinity, uniformly in the distribution of
independent SPD H. It does NOT prove finite moments of order>=5, nor t5 equality.
The actual reference can be lighter-tailed; this is an upper-moment certificate.

With T=V exp(-Y/2), V independent of errors after the inherited rotation step,
equation(1) yields for 0<alpha<min(5,4Nt):

    E[T_+^alpha | targetbranch,s] / E[T_+^alpha]
      >= piT(s+4atanh(alpha/(6Nt)))/piT(s).             (2)

It links how quickly the rule changes from borrowing to target-only across
slack to a cost paid by its selected target reference. It is not just a statement
that selection density is tilted. It is still based on established MLR/tilting
tools; importance/priority require comparison, not rhetorical relabeling.

## Finite transition corollary

For L>0, d=4atanh(alpha/(6Nt)), k=ceil(L/d), let s_j=jL/k,j=0,...,k-1.
Monotonicity of piT and telescoping show that at least one j satisfies

    moment_ratio(s_j) >= [piT(L)/piT(0)]^(1/k).

Proof: d>=L/k, and product_j piT(s_j+d)/piT(s_j)
>= product_j piT(s_{j+1})/piT(s_j)=piT(L)/piT(0).
No tail-probability or Power lower bound follows without another argument.
This is a statement within the selected-target construction, not minimax over
joint-budget alternatives, and never excuses dropping mandatory wideD cases.

## M002 fixed deterministic check, before output

Use D011 all10cases earliestrep8 and frozenW/meta/gate, asD012. No new data or
randomnumbers. Compute piT(s) and exact exponential-tilt representation of (1)
at s=0,.5,1,log5; r=.5,1,2. Gauss-Legendre512 versus1024 nodes in each bank;
separate tilted density normalizers/tangent tail checks. Report numerical gaps,
not empiricalFDR or confidence intervals. Include Gaussian constant-curvature
identity as a unit fixture. Stop at these fixed combinations regardless of sign.
If formula or numerical check fails, diagnose it rather than lower G2.
No main candidate modification, no additional observation batch or confirmation.
