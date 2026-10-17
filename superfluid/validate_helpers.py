"""Helpers for rgengine validation: 't Hooft-coupling transforms of exact beta polynomials."""
from fractions import Fraction
import numpy as np
import rgengine as rg


def thooft(polys, names, gauge_names, scale, N_for_gauge):
    """Transform hatted-coupling polynomials to 't Hooft couplings.
    scale[name] = k with chat = k * c for non-gauge couplings (e.g. hhat = h/N -> k = 1/N).
    gauge: ghat^2 = lam/Ng  (Ng = N_for_gauge[name]); beta_lam = 2 Ng ghat beta_ghat.
    Returns {beta_name: {tuple(exponents in the SAME variable order, gauge exponent counts lam
    powers): float}}."""
    out = {}
    for j, (bn, d) in enumerate(zip(names, polys)):
        res = {}
        for mon, cf in d.items():
            cf = float(cf)
            mon = list(mon)
            if bn in gauge_names:
                mon[j] += 1  # multiply by ghat
                cf *= 2 * N_for_gauge[bn]
            else:
                cf /= scale[bn]
            newmon = []
            for i, (vn, e) in enumerate(zip(names, mon)):
                if vn in gauge_names:
                    assert e % 2 == 0, (bn, mon)
                    cf *= N_for_gauge[vn] ** (-e / 2)
                    newmon.append(e // 2)
                else:
                    cf *= scale[vn] ** e
                    newmon.append(e)
            key = tuple(newmon)
            res[key] = res.get(key, 0.0) + cf
        out[bn] = {k: v for k, v in res.items() if abs(v) > 1e-12}
    return out


def fmt(d, names):
    items = sorted(d.items(), key=lambda kv: kv[0], reverse=True)
    parts = []
    for m, c in items:
        f = Fraction(c).limit_denominator(10000)
        cs = str(f) if abs(float(f) - c) < 1e-9 * max(1, abs(c)) else f"{c:.10g}"
        mon = "*".join(f"{n}^{e}" if e > 1 else n for n, e in zip(names, m) if e > 0)
        parts.append(f"({cs}){mon}")
    return " + ".join(parts)
