"""
table1.py -- reproduces Table I of arXiv:2610.10169 ("A Superfluid at All Temperatures").

For each theory (N1, Nf1, N2, Nf2) it computes the fixed points of the one- and two-loop beta functions, exact in N_i
(family_fixed_points.solve_theory), and the columns of Table I:

    10^3 lambda_1*, 10^3 lambda_2*   two-loop Banks-Zaks couplings
    Delta                            kappa * (lambda_1* - lambda_2*) N_2^2 / lambda_bar
    theta^1L, theta^2L               arg[(f_1 - f_2)/2 + i zeta] at the one- and two-loop fixed points
    c_2^2L / lambda_2*               one-loop thermal mass of Phi_2, evaluated with the two-loop couplings
    D / D_LO                         depth of the ordered minimum including the static modes of the large sector,
                                     relative to its leading-order value (End Matter, "Thermal corrections")

D / D_LO: along the full-rank direction Phi_2 = v 1, with w = v^2/(N_1 T^2) and r = N_2/N_1,

    Delta f(w) / (16 pi^2 N_1^2 T^4) = a_2 w^2 + c_2 r w + (2/3) [ c_1^(3/2) - (c_1 + 2 zeta w)^(3/2) ],

where a_2 = h_2 + f_2. The first two terms are the leading-order potential; the last is the free energy of the static
modes of the 2 N_1^2 real scalars of the large sector, whose mass squared is 16 pi^2 T^2 (c_1 + 2 zeta w). All couplings
are those of the two-loop fixed point, and c_1, c_2 are the full one-loop thermal masses, including the contributions of
the small sector to c_2. D_LO = (c_2 r)^2 / (4 a_2) and D = -min_w Delta f.

Run:  python table1.py      (writes table1.csv and table1_body.tex, and compares with the printed table)
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from family_fixed_points import solve_theory, family_tuple, thermal

HERE = os.path.dirname(os.path.abspath(__file__))

THEORIES = [("(150,783,5,26)", (150, 783, 5, 26), None), ("(300,1562,10,52)", (300, 1562, 10, 52), None)] + \
           [(f"T_{n}", family_tuple(n, 1, 1, -12), n) for n in (5, 9, 13, 21, 41, 101)]

# Body of Table I as printed in arXiv:2610.10169v1
PRINTED = r"""
$(150,783,5,26)$ & 1.468 & 2.517 & $-33.47$ & $-1.844$ & $-1.700$ & $-0.793$ & 0.221\\
$(300,1562,10,52)$ & 2.130 & 2.476 & $-38.25$ & $-1.957$ & $-1.828$ & $-1.79$ & 0.261\\
\hline
$\mathcal{T}_{5}$ & 1.271 & 2.517 & $-41.85$ & $-2.230$ & $-2.143$ & $-0.588$ & 0.200\\
$\mathcal{T}_{9}$ & 1.156 & 1.368 & $-34.60$ & $-1.796$ & $-1.728$ & $-5.76$ & 0.477\\
$\mathcal{T}_{13}$ & 0.871 & 0.941 & $-33.20$ & $-1.725$ & $-1.678$ & $-13.9$ & 0.555\\
$\mathcal{T}_{21}$ & 0.564 & 0.580 & $-32.43$ & $-1.687$ & $-1.659$ & $-39.2$ & 0.641\\
$\mathcal{T}_{41}$ & 0.294 & 0.296 & $-32.06$ & $-1.670$ & $-1.655$ & $-155$ & 0.735\\
$\mathcal{T}_{101}$ & 0.120 & 0.120 & $-31.93$ & $-1.664$ & $-1.658$ & $-951$ & 0.827\\
"""


def depth_ratio(couplings, N1, N2, l1, l2):
    """D/D_LO for the couplings (h1, h2, f1, f2, zeta) of an ordered fixed point."""
    h1, h2, f1, f2, z = couplings
    c1, c2 = thermal(couplings, N1, N2, l1, l2)[:2]
    a2, r = h2 + f2, N2 / N1
    b = c2 * r
    assert c2 < 0 and z < 0 and a2 > 0 and c1 > 0, "not an ordered fixed point"
    w_lo = -b / (2 * a2)
    D_lo = b * b / (4 * a2)
    F = lambda w: a2 * w * w + b * w + 2 / 3 * (c1 ** 1.5 - (c1 + 2 * z * w) ** 1.5)
    dF = lambda w: 2 * a2 * w + b - 2 * z * np.sqrt(c1 + 2 * z * w)
    if dF(0.0) >= 0:          # the correction removes the ordered minimum
        return 0.0
    w = brentq(dF, 0.0, w_lo, xtol=1e-300, rtol=1e-15)
    assert c1 + 2 * z * w > 0
    return -F(w) / D_lo


def compute():
    rows = []
    for label, (N1, Nf1, N2, Nf2), n in THEORIES:
        s = solve_theory(N1, Nf1, N2, Nf2, n=n)
        assert s["ordered_1L"] and s["ordered_2L"], label
        rows.append(dict(theory=label, N1=N1, Nf1=Nf1, N2=N2, Nf2=Nf2,
                         lam1_e3=1e3 * s["lam1"], lam2_e3=1e3 * s["lam2"], Delta=s["Delta"],
                         theta_1L=s["theta_1L"], theta_2L=s["theta_2L"], c2_2L=s["c2_2L"],
                         D_over_DLO=depth_ratio(s["couplings_2L"], N1, N2, s["lam1"], s["lam2"]),
                         res1=s["res1"], res2=s["res2"]))
    return pd.DataFrame(rows)


def sig3(x):
    """three significant figures, as for c_2^2L/lambda_2* in the paper"""
    d = max(0, 2 - int(np.floor(np.log10(abs(x)))))
    return f"{x:.{d}f}"


def latex_rows(df):
    out = []
    for i, r in df.iterrows():
        name = f"${r.theory}$" if r.theory.startswith("(") else f"$\\mathcal{{T}}_{{{r.theory[2:]}}}$"
        m = lambda x, fmt: (f"${x:{fmt}}$" if x < 0 else f"{x:{fmt}}")
        out.append(f"{name} & {r.lam1_e3:.3f} & {r.lam2_e3:.3f} & {m(r.Delta, '.2f')} & {m(r.theta_1L, '.3f')} & "
                   f"{m(r.theta_2L, '.3f')} & ${sig3(r.c2_2L)}$ & {r.D_over_DLO:.3f}\\\\")
        if i == 1:
            out.append("\\hline")
    return "\n".join(out) + "\n"


def compare(df):
    """compare every entry with the printed table, at the printed number of decimals"""
    ok, rows = True, [l for l in PRINTED.strip().splitlines() if "&" in l]
    cols = ["lam1_e3", "lam2_e3", "Delta", "theta_1L", "theta_2L", "c2_2L", "D_over_DLO"]
    for line, (_, r) in zip(rows, df.iterrows()):
        cells = [c.replace("$", "").replace("\\\\", "").strip() for c in line.split("&")]
        for col, s in zip(cols, cells[1:]):
            dec = len(s.split(".")[1]) if "." in s else 0
            if abs(r[col] - float(s)) > 0.5 * 10 ** (-dec) + 1e-12:
                ok = False
                print(f"MISMATCH {r.theory} {col}: computed {r[col]:.6g}, printed {s}")
    return ok


if __name__ == "__main__":
    df = compute()
    df.to_csv(os.path.join(HERE, "table1.csv"), index=False)
    with open(os.path.join(HERE, "table1_body.tex"), "w") as f:
        f.write(latex_rows(df))
    print(df[["theory", "lam1_e3", "lam2_e3", "Delta", "theta_1L", "theta_2L", "c2_2L", "D_over_DLO"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("max residuals: one loop %.1e, two loop %.1e" % (df.res1.max(), df.res2.max()))
    print("agrees with the printed Table I:", compare(df))
