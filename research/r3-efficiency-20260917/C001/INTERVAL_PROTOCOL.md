# Fixed confirmation uncertainty rule

Adopts the classical bounded-mean capital process, not a new R3 method:
[Waudby-Smith–Ramdas, Sec4/Prop2 and AppendixB.6](https://arxiv.org/html/2010.09686).
Those portions were read directly. For independent X_i in[0,1] with mean<=u,
each product Π_i[1+c(X_i/u-1)], fixed0<c<1, is nonnegative with expectation<=1.
The fixed uniform mixture over12 listed fractions also has expectation<=1.
Markov's inequality bounds rejection at1/delta bydelta. Wealth decreases in u,
so inversion gives a lower endpoint; apply the same construction to1-X for
an upper endpoint. Rescale known bounded metrics first. This proves the exact
ideal-arithmetic coverage used here, not just a variance approximation.

The384 TWO-SIDED interval budget uses totalalpha.025; each tail gets
.025/(2*384), giving joint coverage at least97.5%. Fractions are frozen before
confirmation. Bisection includes an outward1e-12 guard; no universal floating
certificate is asserted. Individual model FDR remainsq=.05: it is not this
Monte Carlo reporting error. Whole families are independent units; genes,
folds and components never serve as replications. No optional sample increase.

R2's previous intervals/history are unchanged; this is one prospectively
specified R3 confirmation, not a retrospective re-analysis for narrower R2
claims or a joint confidence statement across all historical research phases.
