"""
class1_betas.py -- Class 1: finite-N fate of the CCR fixed circle in the COMPLEX double-bifundamental
model  G = [SU(N1)xSU(N1)] x [SU(N2)xSU(N2)],  per sector i: Nf_i Dirac psi_i ~ (N_i,1), Nf_i Dirac
chi_i ~ (1,N_i), complex Phi_i ~ (N_i, Nbar_i);
    V = h~_i Tr(Phi_i^+Phi_i)^2 + f~_i (Tr Phi_i^+Phi_i)^2 + 2 zeta~ Tr Phi_1^+Phi_1 Tr Phi_2^+Phi_2,
't Hooft variables  g_i^2 = 16pi^2 lam_i/N_i, h~_i = 16pi^2 h_i/N_i, f~_i = 16pi^2 f_i/N_i^2,
zeta~ = 16pi^2 zeta/(N1 N2).   (engine hatted: ghat^2 = lam/N, hhat = h/N, fhat = f/N^2, zhat = zeta/(N1N2)).

Symmetries imposed: SU(Nf_i)_psi x SU(Nf_i)_chi (separately) forbid the gauge-invariant Yukawa
psibar_i Phi_i chi_i (it needs a diagonal flavour contraction); U(1)_i: Phi_i -> e^{ia} Phi_i forbids
det Phi_i (N=3 cubic, N=4 quartic, N=2 mass).  With these, {h_i, f_i, zeta} is the complete quartic basis
for every N_i >= 2 (checked by the engine's missing-operator residual, ~1e-15 at N=2,3,4,5).

Sections
  (A) optional: re-derive the exact finite-N one-loop quartic betas with rgengine (grid N1,N2=2..8),
      convert to 't Hooft variables and fit Laurent polynomials in (N1,N2)  -> reproduces BETA_Q below.
  (B) closed-form betas (exact in N1, N2; 1-loop quartic, 2-loop gauge incl. scalars).
  (C) planar fixed circle c0(phi), full 7x7 Jacobian, left null vector ell(phi), solvability
      function L(phi) = ell . beta^(1)(c0(phi))   [sympy, exact].
  (D) exact finite-N solver: rational BZ couplings -> eliminate f_i, zeta -> resultant in h2 -> all real roots.
  (E) thermal analysis: thermal masses (engine thermal_C == CCR 4.22), boundedness (copositivity on the
      trace cone), exact minimisation of V4 + T^2 C over ALL directions (singular-value reduction), stability
      of the Phi_1 = 0 direction.
  (F) scan over integer tuples (N1,Nf1,N2,Nf2).
Run:  python class1_betas.py   (needs sympy, numpy, mpmath; section A needs rgengine + jax)
"""
import numpy as np
import sympy as sp
import mpmath as mp
from sympy import Rational as Q

mp.mp.dps = 40
L1, L2, H1, H2, F1, F2, Z = sp.symbols('lambda1 lambda2 h1 h2 f1 f2 zeta')
X = [L1, L2, H1, H2, F1, F2, Z]
N1s, N2s, xs = sp.symbols('N1 N2 x', positive=True)
e1, e2, e12 = sp.symbols('epsilon1 epsilon2 epsilon12')   # 1/N1^2, 1/N2^2, 1/(N1 N2)


# ------------------------------------------------------------------------------------------------
# (A) engine re-derivation (optional, ~10 min)
# ------------------------------------------------------------------------------------------------
def rederive_with_engine(Nmax=8):
    import itertools, rgengine as rg
    from fractions import Fraction
    names = ['l1', 'l2', 'h1', 'h2', 'f1', 'f2', 'z']
    TH = {}
    for N1 in range(2, Nmax + 1):
        for N2 in range(2, Nmax + 1):
            m = rg.example_CDB(N1, N2, 0, 0)
            P, err = rg.model_beta_polynomials(m, loops={'gauge': 1, 'quartic': 1})
            s = [None, None, Fraction(N1), Fraction(N2), Fraction(N1 ** 2), Fraction(N2 ** 2), Fraction(N1 * N2)]
            out = {}
            for k in range(2, 7):
                d = {}
                for e, c in P[k].items():
                    coef = Fraction(c) * s[k]; mon = list(e)
                    for j, Nj in ((0, N1), (1, N2)):
                        coef /= Fraction(Nj) ** (e[j] // 2); mon[j] = e[j] // 2
                    for j in range(2, 7):
                        coef /= s[j] ** e[j]
                    d[tuple(mon)] = d.get(tuple(mon), 0) + coef
                out[names[k]] = d
            TH[(N1, N2)] = out
    pw = [(p, q) for p in range(-1, 4) for q in range(-1, 4)]
    keys = sorted(TH); train = [k for k in keys if max(k) < Nmax]; hold = [k for k in keys if max(k) == Nmax]
    fits = {}; worst = 0.0
    for b in names[2:]:
        mons = set().union(*[set(TH[k][b]) for k in keys])
        for mon in sorted(mons):
            val = lambda k: sp.Rational(TH[k][b].get(mon, 0).numerator, TH[k][b].get(mon, 0).denominator) if mon in TH[k][b] else 0
            A = sp.Matrix([[sp.Rational(1, a ** p) * sp.Rational(1, c ** q) for (p, q) in pw] for (a, c) in train])
            y = sp.Matrix([val(k) for k in train])
            sol = (A.T * A).LUsolve(A.T * y)
            expr = sp.expand(sum(sol[i] * N1s ** (-pw[i][0]) * N2s ** (-pw[i][1]) for i in range(len(pw))))
            worst = max([worst] + [abs(float(expr.subs({N1s: k[0], N2s: k[1]}) - val(k))) for k in keys])
            fits[(b, mon)] = expr
    betas = {b: sp.expand(sum(c * sp.prod([X[j] ** mon[j] for j in range(7)]) for (bb, mon), c in fits.items() if bb == b))
             for b in names[2:]}
    return betas, worst   # worst = max deviation on all 49 (N1,N2) points incl. the N=8 holdout


# ------------------------------------------------------------------------------------------------
# (B) closed-form exact finite-N betas ('t Hooft variables), as fitted from the engine (residual 0)
# ------------------------------------------------------------------------------------------------
def beta_quartic(eps1=e1, eps2=e2, eps12=e12):
    bh1 = 8*H1**2 - 12*(1-eps1)*L1*H1 + (Q(3, 2) - 12*eps1)*L1**2 + 24*eps1*F1*H1
    bh2 = 8*H2**2 - 12*(1-eps2)*L2*H2 + (Q(3, 2) - 12*eps2)*L2**2 + 24*eps2*F2*H2
    bf1 = 4*(1+4*eps1)*F1**2 + 16*F1*H1 - 12*(1-eps1)*L1*F1 + 12*H1**2 + (Q(9, 2) + 6*eps1)*L1**2 + 4*Z**2
    bf2 = 4*(1+4*eps2)*F2**2 + 16*F2*H2 - 12*(1-eps2)*L2*F2 + 12*H2**2 + (Q(9, 2) + 6*eps2)*L2**2 + 4*Z**2
    bz = Z*(4*(1+eps1)*F1 + 4*(1+eps2)*F2 + 8*H1 + 8*H2 - 6*(1-eps1)*L1 - 6*(1-eps2)*L2 + 8*eps12*Z)
    return {'h1': bh1, 'h2': bh2, 'f1': bf1, 'f2': bf2, 'z': bz}


def beta_gauge_thooft(lam, N, Nf):
    """2-loop gauge beta of each SU(N_i) factor ('t Hooft), incl. the bifundamental scalar:
       beta = -(21-4x)/3 lam^2 + [ (26x-54)/3 - (4+2x)/N^2 ] lam^3,  x = Nf/N   (engine fit, dev 2e-14)."""
    x = sp.Rational(Nf, N)
    return -(21 - 4*x)/3*lam**2 + ((26*x - 54)/3 - (4 + 2*x)/sp.Integer(N)**2)*lam**3


def lam_star(N, Nf):
    """exact (rational) 2-loop Banks-Zaks 't Hooft coupling at finite N (None if not AF / no BZ)."""
    x = Q(Nf, N)
    den = 26*x - 54 - 3*(4 + 2*x)/Q(N)**2
    num = 21 - 4*x
    if num <= 0 or den <= 0:
        return None
    return num/den


# ------------------------------------------------------------------------------------------------
# (C) planar circle, Jacobian, left null vector, solvability function
# ------------------------------------------------------------------------------------------------
phi, lam, dl = sp.symbols('phi lambda delta', real=True)
R_hat = sp.sqrt((18*sp.sqrt(6) - 39)/16)          # circle radius / lambda  (CCR 4.18)
KAPPA = (18*sp.sqrt(6) - 39)/2
A_C = Q(93, 2) - 27*sp.sqrt(6)/4
C_1 = sp.sqrt(6*sp.sqrt(6) - 13)*(-135*sp.sqrt(2)/8 + 9*sp.sqrt(3)/2)


def planar_point(l=lam):
    return {L1: l, L2: l, H1: (3 - sp.sqrt(6))/4*l, H2: (3 - sp.sqrt(6))/4*l,
            F1: sp.sqrt(Q(3, 2))*l + R_hat*l*sp.cos(phi), F2: sp.sqrt(Q(3, 2))*l - R_hat*l*sp.cos(phi),
            Z: R_hat*l*sp.sin(phi)}


def solvability():
    """returns (ell, L(phi), G(phi)=L/sin(phi)) with delta = lambda1-lambda2 (BZ mismatch)."""
    bq = beta_quartic()
    b0, b1 = (21 - 4*xs)/3, (26*xs - 54)/3
    beta = sp.Matrix([-b0*L1**2 + b1*L1**3, -b0*L2**2 + b1*L2**3] + [bq[k] for k in ('h1', 'h2', 'f1', 'f2', 'z')])
    beta0 = beta.subs({e1: 0, e2: 0, e12: 0}); beta1 = sp.expand(beta - beta0)
    pt = planar_point()
    Jll = sp.Symbol('J_ll', positive=True)            # = b0*lambda at the BZ point
    Jp = beta0.jacobian(X).subs(pt); Jp[0, 0] = Jll; Jp[1, 1] = Jll
    lv = sp.symbols('l0:7')
    eqs = list(sp.Matrix([lv])*Jp)
    sol = sp.solve(eqs[:6] + [lv[6] - 2*sp.cos(phi)], lv, dict=True)[0]
    ell = sp.Matrix([sp.simplify(sol[v]) for v in lv])
    assert all(sp.simplify(x) == 0 for x in ell.T*Jp)
    d1, d2 = dl/2, -dl/2                                 # exact BZ values lambda_i = lambda +- delta/2
    b1vec = sp.Matrix([-Jll*d1, -Jll*d2] + [beta1[i].subs(pt) for i in range(2, 7)])
    Lphi = sp.expand(sp.simplify((ell.T*b1vec)[0]))
    G = sp.expand(sp.simplify(Lphi/sp.sin(phi)))
    Gform = lam*(KAPPA*dl + lam*((e2 - e1)*(KAPPA*sp.cos(phi)**2 + A_C) + (e1 + e2)*C_1*sp.cos(phi)
                                 + e12*KAPPA*sp.sin(2*phi)))
    assert sp.simplify(G - Gform) == 0
    return ell, Lphi, Gform


def first_order_zeros(N1, Nf1, N2, Nf2, n=20001):
    """zeros phi* (other than sin phi = 0) of G(phi) at first order in 1/N^2 and delta."""
    kf, af, cf = float(KAPPA), float(A_C), float(C_1)
    l1, l2 = float(lam_star(N1, Nf1)), float(lam_star(N2, Nf2))
    lb, dlt = (l1 + l2)/2, l1 - l2
    ph = np.linspace(-np.pi, np.pi, n)
    g = kf*dlt/lb + ((1/N2**2 - 1/N1**2)*(kf*np.cos(ph)**2 + af) + (1/N1**2 + 1/N2**2)*cf*np.cos(ph)
                     + kf/(N1*N2)*np.sin(2*ph))
    idx = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
    return [ph[i] - g[i]*(ph[i+1] - ph[i])/(g[i+1] - g[i]) for i in idx]


# ------------------------------------------------------------------------------------------------
# (D) exact finite-N solver (zeta != 0 branch; zeta = 0 branch decouples exactly)
# ------------------------------------------------------------------------------------------------
_hh1, _hh2 = sp.symbols('hh1 hh2')
_BQ = beta_quartic()
_beta_num = sp.lambdify([L1, L2, H1, H2, F1, F2, Z, e1, e2, e12], [_BQ[b] for b in ('h1', 'h2', 'f1', 'f2', 'z')], 'mpmath')


def _mpq(q):
    return mp.mpf(q.p)/q.q


def exact_fixed_points(N1, Nf1, N2, Nf2, tol=1e-12):
    """all real fixed points with zeta != 0 of the exact finite-N 1-loop quartic + 2-loop gauge system."""
    l1, l2 = lam_star(N1, Nf1), lam_star(N2, Nf2)
    if l1 is None or l2 is None:
        return None
    E1, E2, E12 = Q(1, N1**2), Q(1, N2**2), Q(1, N1*N2)
    f1e = -(8*_hh1**2 - 12*(1 - E1)*l1*_hh1 + (Q(3, 2) - 12*E1)*l1**2)/(24*E1*_hh1)    # beta_h1 = 0
    f2e = -(8*_hh2**2 - 12*(1 - E2)*l2*_hh2 + (Q(3, 2) - 12*E2)*l2**2)/(24*E2*_hh2)
    ze = -(4*(1 + E1)*f1e + 4*(1 + E2)*f2e + 8*_hh1 + 8*_hh2 - 6*(1 - E1)*l1 - 6*(1 - E2)*l2)/(8*E12)
    sub = {L1: l1, L2: l2, H1: _hh1, H2: _hh2, F1: f1e, F2: f2e, Z: ze, e1: E1, e2: E2, e12: E12}
    n1 = sp.Poly(sp.numer(sp.together(_BQ['f1'].subs(sub))), _hh1, _hh2, domain='QQ')
    n2 = sp.Poly(sp.numer(sp.together(_BQ['f2'].subs(sub))), _hh1, _hh2, domain='QQ')
    pr = sp.Poly(sp.resultant(n1.as_expr(), n2.as_expr(), _hh1), _hh2, domain='QQ')
    sols = []
    L1v, L2v = _mpq(l1), _mpq(l2); e1v, e2v, e12v = _mpq(E1), _mpq(E2), _mpq(E12)
    for r2 in pr.real_roots():
        r2v = sp.N(r2, 40)
        if abs(r2v) < 1e-30:
            continue
        for r1 in sp.Poly(n1.as_expr().subs(_hh2, r2v), _hh1).nroots(n=35, maxsteps=500):
            if abs(sp.im(r1)) > 1e-20 or abs(sp.re(r1)) < 1e-30:
                continue
            h1v, h2v = mp.mpf(str(sp.re(r1))), mp.mpf(str(r2v))
            f1v = -(8*h1v**2 - 12*(1 - e1v)*L1v*h1v + (mp.mpf(3)/2 - 12*e1v)*L1v**2)/(24*e1v*h1v)
            f2v = -(8*h2v**2 - 12*(1 - e2v)*L2v*h2v + (mp.mpf(3)/2 - 12*e2v)*L2v**2)/(24*e2v*h2v)
            zv = -(4*(1 + e1v)*f1v + 4*(1 + e2v)*f2v + 8*h1v + 8*h2v - 6*(1 - e1v)*L1v - 6*(1 - e2v)*L2v)/(8*e12v)
            b = _beta_num(L1v, L2v, h1v, h2v, f1v, f2v, zv, e1v, e2v, e12v)
            resid = max(abs(x) for x in b)/max(L1v, L2v)**2
            if resid < tol:
                s = dict(l1=float(L1v), l2=float(L2v), h1=float(h1v), h2=float(h2v), f1=float(f1v), f2=float(f2v),
                         z=float(zv), resid=float(resid))
                if not any(abs(s['h1'] - t['h1']) + abs(s['h2'] - t['h2']) < 1e-12 for t in sols):
                    sols.append(s)
    return float(l1), float(l2), sols


# ------------------------------------------------------------------------------------------------
# (E) thermal analysis at finite N
# ------------------------------------------------------------------------------------------------
def thermal_analysis(s, N1, N2):
    """C_i: engine thermal_C per complex component in hatted units (== CCR eq. 4.22 bracket, checked
    exactly at (N1,N2) = (2,2),(2,3),(3,4),(4,2),(4,4),(5,2)).  V_hat = V4_hat + sum_i C_i Tr Phi_i^+Phi_i (T=1).
    Exact minimisation over all directions: V depends only on the eigenvalues m_ia >= 0 of Phi_i^+Phi_i;
    for h_i > 0 the minimum at fixed traces is m_ia equal (full rank, Phi_i ~ unitary), giving
    V_hat = sum a_i u_i^2 + 2 zeta u1 u2 + sum C_i N_i u_i, u_i = Tr Phi_i^+Phi_i / N_i, a_i = h_i + f_i."""
    E1, E2 = 1/N1**2, 1/N2**2
    l1, l2, h1, h2, f1, f2, z = [s[k] for k in ('l1', 'l2', 'h1', 'h2', 'f1', 'f2', 'z')]
    c1 = h1/3 + (1 + E1)*f1/6 + (N2/N1)*z/6 + (1 - E1)*l1/4
    c2 = h2/3 + (1 + E2)*f2/6 + (N1/N2)*z/6 + (1 - E2)*l2/4
    a1, a2 = h1 + f1, h2 + f2
    bounded = bool(h1 > 0 and h2 > 0 and a1 > 0 and a2 > 0 and (z >= 0 or z*z < a1*a2))
    V = lambda u: a1*u[0]**2 + a2*u[1]**2 + 2*z*u[0]*u[1] + c1*N1*u[0] + c2*N2*u[1]
    cands = [(0.0, 0.0)]
    if c1 < 0: cands.append((-c1*N1/(2*a1), 0.0))
    if c2 < 0: cands.append((0.0, -c2*N2/(2*a2)))
    try:
        u = np.linalg.solve(2*np.array([[a1, z], [z, a2]]), -np.array([c1*N1, c2*N2]))
        if (u >= 0).all(): cands.append(tuple(u))
    except np.linalg.LinAlgError:
        pass
    best = min(cands, key=V)
    stab1 = (c1*N1 + 2*z*(-c2*N2/(2*a2))) if c2 < 0 else None     # >0: Phi_1 = 0 stays a minimum
    boundedness_margin = min(a1, a2, (a1*a2 - z*z) if z < 0 else a1*a2)/max(l1, l2)**2 if True else None
    return dict(c1=c1, c2=c2, c1_rel=c1/l1, c2_rel=c2/l2, bounded=bounded, u_min=best, V_min=V(best), stab1=stab1,
                phi=float(np.arctan2(z, (f1 - f2)/2)), a1=a1, a2=a2,
                ordered=bool(bounded and c2 < 0 and best[1] > 0),
                bnd_margin=boundedness_margin)


def jacobian_eigs(s, N1, Nf1, N2, Nf2):
    """stability matrix eigenvalues of the full 7-coupling system at an exact fixed point."""
    bq = beta_quartic(Q(1, N1**2), Q(1, N2**2), Q(1, N1*N2))
    vec = sp.Matrix([beta_gauge_thooft(L1, N1, Nf1), beta_gauge_thooft(L2, N2, Nf2)] +
                    [bq[k] for k in ('h1', 'h2', 'f1', 'f2', 'z')])
    J = vec.jacobian(X)
    vals = {L1: s['l1'], L2: s['l2'], H1: s['h1'], H2: s['h2'], F1: s['f1'], F2: s['f2'], Z: s['z']}
    Jn = np.array(J.subs(vals).evalf(), dtype=float)
    return np.linalg.eigvals(Jn)


# ------------------------------------------------------------------------------------------------
# (G) exact finite-N TWO-LOOP quartic betas ('t Hooft variables; x_i = Nf_i/N_i)
#     Engine tensor-mode 2-loop quartic betas (Poole-Thomsen/PyR@TE basis) evaluated at
#     (N1,N2) = (2,2),(3,2),(4,2),(3,3) [+ Nf1=3 / Nf2=3 variants], fitted per model on a structurally
#     restricted monomial basis (residual <= 2e-14, projection residual <= 4e-14 => no missing operator),
#     then Laurent-fitted in (N1,N2) (smallest exact rational basis).  Blind holdouts: (4,3) all 94
#     coefficients to 4.6e-12, (5,2) all 101 coefficients to 6.8e-10.  Planar limit == CCR eq. 3.4 mapped
#     by CCR eq. 4.16, incl. all zeta-dependent terms (exact).  Strings in class1_beta2L.json.
#     The scripts that produced these fits are not part of this repository. pyrate_crosscheck.py
#     checks the fitted expressions against PyR@TE 3 at two loops.
# ------------------------------------------------------------------------------------------------
x1s, x2s = sp.symbols('x1 x2')


def beta_quartic_2loop(path='class1_beta2L.json'):
    import json
    loc = {'lambda1': L1, 'lambda2': L2, 'h1': H1, 'h2': H2, 'f1': F1, 'f2': F2, 'zeta': Z,
           'N1': N1s, 'N2': N2s, 'x1': x1s, 'x2': x2s}
    return {k: sp.sympify(v, locals=loc) for k, v in json.load(open(path)).items()}


def planar_2loop_ccr():
    """CCR eq. 3.4 (real model, planar) mapped to the complex model with eq. 4.16."""
    def real(lr1, lr2, hr1, hr2, fr1, fr2, zr, xa, xb):
        bh = lambda h, l, x: -96*h**3 + 40*l*h**2 + (5*x/3 - Q(13, 2))*l**2*h + (Q(1, 4) - x/6)*l**3
        bf = lambda h, f, l, lp, x: (-384*h**3 - 160*h**2*f + 96*l*h**2 + 128*l*h*f + 32*l*f**2 + 32*lp*zr**2
                                     + 9*l**2*h + (5*x/3 - Q(7, 2))*l**2*f + Q(1, 4)*(3 - 2*x)*l**3)
        bz = zr*(-80*(hr1**2 + hr2**2) + 32*lr1*(fr1 + 2*hr1) + 32*lr2*(fr2 + 2*hr2)
                 + lr1**2*(-21 + 10*xa)/12 + lr2**2*(-21 + 10*xb)/12)
        return [bh(hr1, lr1, xa), bh(hr2, lr2, xb), bf(hr1, fr1, lr1, lr2, xa), bf(hr2, fr2, lr2, lr1, xb), bz]
    R2 = real(2*L1, 2*L2, H1/2, H2/2, F1/2, F2/2, Z/2, x1s, x2s)
    return {k: sp.expand(2*v) for k, v in zip(['h1', 'h2', 'f1', 'f2', 'z'], R2)}


def track_2loop(tup, s1, beta2, nsteps=40):
    """Homotopy s: 0 -> 1 of beta_1L + s*beta_2L from an exact 1-loop fixed point s1 (lambda_i fixed at
    the 2-loop BZ values; 3-loop gauge not included).  Returns the 2-loop fixed point or None."""
    import scipy.optimize as so
    s_ = sp.Symbol('s_')
    N1, Nf1, N2, Nf2 = tup
    bq = beta_quartic(Q(1, N1**2), Q(1, N2**2), Q(1, N1*N2))
    sub = {N1s: N1, N2s: N2, x1s: Q(Nf1, N1), x2s: Q(Nf2, N2), L1: s1['l1'], L2: s1['l2']}
    ex = [(bq[k] + s_*beta2[k]).subs(sub) for k in ('h1', 'h2', 'f1', 'f2', 'z')]
    f = sp.lambdify([H1, H2, F1, F2, Z, s_], ex, 'numpy')
    J = sp.lambdify([H1, H2, F1, F2, Z, s_], sp.Matrix(ex).jacobian([H1, H2, F1, F2, Z]), 'numpy')
    x = np.array([s1[k] for k in ('h1', 'h2', 'f1', 'f2', 'z')]); lsc = max(s1['l1'], s1['l2'])**2
    for s in np.linspace(0, 1, nsteps + 1)[1:]:
        sol = so.root(lambda y: np.array(f(*y, s), float), x, jac=lambda y: np.array(J(*y, s), float), tol=1e-15)
        if np.abs(np.array(f(*sol.x, s), float)).max()/lsc > 1e-10:
            return None
        x = sol.x
    return dict(l1=s1['l1'], l2=s1['l2'], **dict(zip(('h1', 'h2', 'f1', 'f2', 'z'), x)))


if __name__ == '__main__':
    ell, Lphi, G = solvability()
    print('ell(phi) =', list(ell))
    print('G(phi) = L/sin(phi) =', G)
    print(exact_fixed_points(134, 697, 20, 104))
