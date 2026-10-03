# Exact negative control: selecting a smaller scale invalidates a fixed law

This is a CLASSICAL selection-bias counterexample, not an R5 novelty claim.
It supplements D000, whose small shared-reference rates did not prove inflation.

Take the allowed model with D=delta=1, Ns=Nt=N>=4 and a central signed scalar
pivot V, independent of bank errors, symmetric with a positive continuous
density on the positive halfline. The R4 shape/radial pivot has these properties.
Errors Es,Et are iid centered means of logF42, nondegenerate and continuous.
Choose the source scale when Ks<=Kt, otherwise target scale. This selector uses
only available banks; its chosen error is Emin=min(Es,Et).

Let c_alpha>0 be the upperalpha quantile of V*exp(-Et/2), alpha∈(0,.5).
If one incorrectly uses that FIXED-bank reference after selection, the actual
central rejection probability is

 E[tail_V(c_alpha*exp(Emin/2))]
 > E[tail_V(c_alpha*exp(Et/2))] = alpha.

Proof: Emin<=Et pointwise; on the positive-probability event Es<Et it is strict.
For a positive argument, tail_V(c*exp(e/2)) is strictly decreasing ine. Integrate.
Thus the single-test fixed-reference null law is wrong after a calibration-only
choice, even with noTEST access and no true nuisance input. Equal Ns/Nt means
there is no ambiguity about which fixed reference one mistakenly reused.

This is an analytic population-reference counterexample, not a claimed measured
FDR of the complete conservative PC/eBH system, nor a finite-M numerical size
estimate. It pinpoints the broken link that complete selection calibration repairs.
