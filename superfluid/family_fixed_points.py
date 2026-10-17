import numpy as np
import json
import sympy as sp
from scipy.optimize import fsolve
from fractions import Fraction as Fr
import pandas as pd
import os
import warnings

k = 16 * np.pi ** 2

kap = (18 * np.sqrt(6) - 39) / 2
a_ = 93 / 2 - 27 * np.sqrt(6) / 4
b_ = np.sqrt(6 * np.sqrt(6) - 13) * (9 * np.sqrt(3) / 2 - 135 * np.sqrt(2) / 8)
G0 = 43.5 / 82.5
R0 = np.sqrt((18 * np.sqrt(6) - 39) / 16)


def lam_star(N, Nf):
    x = Fr(Nf, N)
    return float((21 - 4 * x) / (26 * x - 54 - 3 * (4 + 2 * x) / Fr(N * N)))


def b1(c, N1, N2, l1, l2):
    h1, h2, f1, f2, z = c
    e1, e2, e12 = 1 / N1 ** 2, 1 / N2 ** 2, 1 / (N1 * N2)
    bh = lambda h, f, l, e: 8 * h * h - 12 * (1 - e) * l * h + (1.5 - 12 * e) * l * l + 24 * e * f * h
    bf = lambda h, f, l, e: 4 * (1 + 4 * e) * f * f + 16 * f * h - 12 * (1 - e) * l * f + 12 * h * h + (4.5 + 6 * e) * l * l + 4 * z * z
    bz = z * (4 * (1 + e1) * f1 + 4 * (1 + e2) * f2 + 8 * h1 + 8 * h2 - 6 * (1 - e1) * l1 - 6 * (1 - e2) * l2 + 8 * e12 * z)
    return np.array([bh(h1, f1, l1, e1), bh(h2, f2, l2, e2), bf(h1, f1, l1, e1), bf(h2, f2, l2, e2), bz])


J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'class1_beta2L.json')))
S = sp.symbols("lambda1 lambda2 h1 h2 f1 f2 zeta N1 N2 x1 x2")
loc = {str(s): s for s in S}
B2f = [sp.lambdify(S, sp.sympify(J[k_], locals=loc), "numpy") for k_ in ["h1", "h2", "f1", "f2", "z"]]


def b2(c, N1, N2, l1, l2, x1, x2):
    h1, h2, f1, f2, z = c
    return np.array([f(l1, l2, h1, h2, f1, f2, z, N1, N2, x1, x2) for f in B2f])


def family_tuple(n, c, m2, j):
    q = c * n * n
    N1 = q * n
    m1 = q * m2 + j
    assert (21 * n - m2) % 4 == 0 and (21 * N1 - m1) % 4 == 0, (n, m2, j)
    return N1, (21 * N1 - m1) // 4, n, (21 * n - m2) // 4


def lo_theta(Delta):
    A, B, C = kap, b_, a_ + Delta
    disc = B * B - 4 * A * C
    u = (-B - np.sqrt(disc)) / (2 * A) if abs((-B - np.sqrt(disc)) / (2 * A)) <= 1 else (-B + np.sqrt(disc)) / (2 * A)
    return -np.arccos(u)


def thermal(c, N1, N2, l1, l2):
    h1, h2, f1, f2, z = c
    e1, e2 = 1 / N1 ** 2, 1 / N2 ** 2
    c1 = h1 / 3 + (1 + e1) * f1 / 6 + (N2 / N1) * z / 6 + (1 - e1) * l1 / 4
    c2 = h2 / 3 + (1 + e2) * f2 / 6 + (N1 / N2) * z / 6 + (1 - e2) * l2 / 4
    a1, a2 = h1 + f1, h2 + f2
    bounded = h1 > 0 and h2 > 0 and a1 > 0 and a2 > 0 and (z >= 0 or z * z < a1 * a2)
    u2 = -c2 * N2 / (2 * a2) if c2 < 0 else 0.0
    stab1 = c1 * N1 + 2 * z * u2
    return c1, c2, bounded, u2, stab1


RES_TOL = 1e-10   # maximal residual of the beta functions at a fixed point, in units of lambda_bar^2


def solve_theory(N1, Nf1, N2, Nf2, twoloop=True, n=None):
    """One- and two-loop fixed point near the ordered arc for the theory (N1, Nf1, N2, Nf2)."""
    l1, l2 = lam_star(N1, Nf1), lam_star(N2, Nf2)
    lb = (l1 + l2) / 2
    d = l1 - l2
    Dex = kap * d * N2 ** 2 / lb
    th = lo_theta(Dex)
    h = (3 - np.sqrt(6)) / 4
    guess = np.array([h * l1, h * l2, np.sqrt(1.5) * lb + R0 * lb * np.cos(th), np.sqrt(1.5) * lb - R0 * lb * np.cos(th), R0 * lb * np.sin(th)])
    F1 = lambda y: b1(y * lb, N1, N2, l1, l2) / lb ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # xtol warnings; residuals are checked below
        y1, info, ier, msg = fsolve(F1, guess / lb, full_output=True, xtol=1e-14)
    c1s = y1 * lb
    res1 = np.abs(F1(y1)).max()
    assert res1 < RES_TOL, f"one-loop fixed point not converged for {(N1, Nf1, N2, Nf2)}: residual {res1:.2e}"
    out = dict(n=n, N1=N1, Nf1=Nf1, N2=N2, Nf2=Nf2, lam1=l1, lam2=l2, Delta=Dex, theta_LO=th,
               theta_1L=np.arctan2(c1s[4], (c1s[2] - c1s[3]) / 2), res1=res1)
    t = thermal(c1s, N1, N2, l1, l2)
    out.update(c1_1L=t[0] / l1, c2_1L=t[1] / l2, bounded_1L=t[2], stab_1L=t[4] / (t[0] * N1),
               ordered_1L=t[2] and t[1] < 0 and t[0] > 0 and t[4] > 0)
    if twoloop:
        x1, x2 = Nf1 / N1, Nf2 / N2
        y = y1.copy()
        for s in np.linspace(0.1, 1, 10):
            Fs = lambda yy: (b1(yy * lb, N1, N2, l1, l2) + s * b2(yy * lb, N1, N2, l1, l2, x1, x2)) / lb ** 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                y = fsolve(Fs, y, xtol=1e-14)
        c2s = y * lb
        out["res2"] = np.abs(Fs(y)).max()
        assert out["res2"] < RES_TOL, f"two-loop fixed point not converged for {(N1, Nf1, N2, Nf2)}: residual {out['res2']:.2e}"
        out["theta_2L"] = np.arctan2(c2s[4], (c2s[2] - c2s[3]) / 2)
        t = thermal(c2s, N1, N2, l1, l2)
        out.update(c1_2L=t[0] / l1, c2_2L=t[1] / l2, bounded_2L=t[2], stab_2L=t[4] / (t[0] * N1),
                   ordered_2L=t[2] and t[1] < 0 and t[0] > 0 and t[4] > 0,
                   zeta_2L=c2s[4], couplings_2L=c2s)
    return out


def solve_member(n, c=1, m2=1, j=-12, twoloop=True):
    """Member T_n of the family N1 = n^3, N2 = n (Eq. (12) of the paper for c=1, m2=1, j=-12)."""
    N1, Nf1, N2, Nf2 = family_tuple(n, c, m2, j)
    return solve_theory(N1, Nf1, N2, Nf2, twoloop=twoloop, n=n)


if __name__ == "__main__":
    rows = [solve_member(n) for n in [5, 9, 13, 17, 21, 25, 29, 41, 61, 101]]
    df = pd.DataFrame(rows)
    fam = df[["n", "N1", "Nf1", "N2", "Nf2", "lam1", "lam2", "Delta", "theta_LO", "theta_1L", "theta_2L",
              "c1_1L", "c2_1L", "c2_2L", "stab_2L", "ordered_1L", "ordered_2L", "res1", "res2"]].copy()
    fam.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "family_fixed_points.csv"), index=False)
    print(fam[["n", "theta_1L", "theta_2L", "c2_2L", "ordered_2L", "res1", "res2"]].to_string(index=False))
