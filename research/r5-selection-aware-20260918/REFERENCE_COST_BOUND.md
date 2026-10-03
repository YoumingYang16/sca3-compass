# Finite reference selection-cost decomposition — classical derived bound

Not a Power bound or an accepted novelty claim. Condition on W and independent
meta. Independent Es,Et have variances Vs,Vt and entropies hs,ht. Let
ds=.5log(2*pi*e*Vs)-hs, dt analogously, the Gaussian-reference entropy deficits.
For L=aEs+(1-a)Et,U=Es-Et, the transform has absolute determinant1, giving
h(L,U)=hs+ht. Marginal Gaussian maximum-entropy bounds imply
I(L;U)<=.5log(Var(L)*Var(U)/(Vs*Vt))+ds+dt=:J.
The variance factor equals1/(1-rho_LU²). Exact inverse-variance a makes rho0;
finite-meta a generally does not, so the Gaussian mismatch term remains.

For B={U<=c}, pi=P(B)>0, data processing and Pinsker give
pi*KL(P_L|B||P_L)+(1-pi)*KL(P_L|notB||P_L)<=I(L;U), hence
TV(P_L|B,P_L)<=min(1,sqrt(J/(2*pi))).
Applying an independent central V and the scale map cannot increase TV.
This bounds the borrowed REFERENCE-law change at boundary, not final finite-MC
PC Power. Bank-dependent DIR under alternatives needs separate handling. Rare
branches or large J can make the bound vacuous; no estimate is used as a validity
certificate. Entropy/data processing/Pinsker are classic tools, not a G1 PASS.

Allzero residual analytic fixture: shifted logBetaPrime(alpha=2N,beta=N) has
variance trigamma(alpha)+trigamma(beta) and entropy
logBeta(alpha,beta)-alpha*digamma(alpha)-beta*digamma(beta)
 +(alpha+beta)*digamma(alpha+beta). Its deficit remains positive at finiteN.
