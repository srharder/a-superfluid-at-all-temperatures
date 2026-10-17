# Entropic order in a lattice model

A classical Monte Carlo illustration of the mechanism in [arXiv:2610.10169](https://arxiv.org/abs/2610.10169): a small
sector orders when heated, because a large reservoir of other fields fluctuates more, and so gains entropy, once the small sector condenses.

**Model.** A classical field theory (the static modes of a thermal field theory) on a periodic $L^3$ cubic lattice with
weight $e^{-H/T}$, a superfluid order parameter $\phi_x\in\mathbb{R}^2$ and a reservoir $\chi_x\in\mathbb{R}^N$, $N=32$:

$$H=\sum_x\Big[\tfrac12\sum_\mu(\phi_{x+\mu}-\phi_x)^2+\tfrac{m^2}{2}\phi_x^2+\tfrac{\lambda}{4}(\phi_x^2)^2+\tfrac12\sum_\mu(\chi_{x+\mu}-\chi_x)^2+\tfrac{M^2}{2}\chi_x^2+\tfrac{g}{4}(\chi_x^2)^2-\tfrac{\kappa}{2}\phi_x^2\chi_x^2\Big],$$

with $\lambda=0.1$, $g=1/N$, $\kappa=0.9\sqrt{\lambda g}$ ($H$ is bounded below for $\kappa^2<\lambda g$), $m^2=0.3$, $M^2=0.5$. The control has $\kappa=0$ and $m^2=-0.3$: an ordinary superfluid, with the reservoir decoupled.

**Mechanism.** A condensate of $\phi$ lowers the mass of the $N$ reservoir fields. In the Hartree approximation the mass of $\phi$ is
$m^2_{\rm eff}(T)\approx m^2+(4\lambda-\kappa N)\,T\,G$, where $G\le0.2527$ is the lattice propagator at the origin. Since
$\kappa N=1.61>4\lambda=0.4$, the thermal correction is negative and grows with $T$; the self-consistent Hartree onset of
order is $T=1.58$. The mechanism goes back to S. Weinberg, Phys. Rev. D 9, 3357 (1974).

**Running.** `python simulate_entropic_order.py` ($L=12$; about 6 minutes on one laptop core) writes `entropic_order_data.npz`,
`entropic_order_scan.csv` and `entropic_order.png/.pdf`; `--quick` ($L=8$; about 40 s) writes the same files with suffix `_quick`;
`--check` tests the energy bookkeeping and free fields (5 s); `--plot-only` redraws the figure. Needs numpy and matplotlib.
The algorithm is checkerboard Metropolis; each model is heated from its ground state from $T=0.2$ to $6$, then cooled back.

**Figure.** (a) The entropic model is disordered at low $T$ (order parameter $\langle|\sum_x\phi_x|\rangle/V=0.03$ at
$T=0.2$, a finite-volume value) and orders at $T\approx1.5$: the Binder cumulant (0 when disordered, 1 when ordered) is 1/2
at $T=1.52$ on heating and $1.46$ on cooling. The order parameter grows to $3.12$ at $T=6$. The control is ordered at low $T$
($1.69$ at $T=0.2$) and melts near $T\approx3.5$–$4$. (b) Once $\phi$ orders, $\langle\chi^2\rangle$ exceeds its value in the
control, by 11% at $T=6$. (c) The phase of $\phi$ on one lattice plane at $T=0.6$ and $T=5$. The cooling scans retrace the
heating scans, so there is no sign of hysteresis; their scatter suggests that the error bars, which include the measured
autocorrelation times, are too small by up to a factor of two.

**Caveat.** The paper's theories are conformal, so order at one temperature implies order at all temperatures. This model
has a lattice scale and shows only the mechanism, order that appears upon heating. At very high $T$ the nearest-neighbour
coupling becomes negligible next to the on-site terms and the order is lost again (on an $8^3$ lattice, between $T=10^3$ and $10^4$).
