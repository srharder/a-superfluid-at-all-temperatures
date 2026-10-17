"""Validation suite for rgengine. Usage: python run_validation.py SECTION [SECTION...]
Writes validation_results_<SECTION>.json.  Sections: ON SM QCD B19 RDB1 CDB1 SO2L RDB2L LS
RESID FUNC VAC FP PERF"""
import json, sys, time, itertools
import numpy as np
import jax.numpy as jnp
import rgengine as rg
import validate_helpers as vh

R = {}


def save(sec):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        return str(o)
    json.dump(R, open(f"validation_results_{sec}.json", "w"), indent=1, default=conv)


def maxdev(ref, got):
    keys = set(ref) | set(got)
    return max(abs(ref.get(k, 0) - got.get(k, 0)) for k in keys) if keys else 0.0


def sec_ON():
    out = []
    for N in range(1, 7):
        m = rg.example_ON(N)
        P, err = rg.model_beta_polynomials(m, loops={"quartic": 2}, split_loops=True)
        b1 = float(P[0][1].get((2,), 0)); b2 = float(P[0][2].get((3,), 0))
        # V = lam (phi^2)^2, lam = lambda/24: expected 8(N+8) lam^2 - 192(3N+14) lam^3
        C = m.thermal_C(np.array([1.0]), raw=False)
        out.append({"N": N, "b1": b1, "b1_ref": 8 * (N + 8), "b2": b2, "b2_ref": -192 * (3 * N + 14),
                    "C_over_lam": C[0, 0], "C_ref": (N + 2) / 3.0, "C_offdiag": float(np.abs(C - C[0, 0] * np.eye(N)).max()),
                    "fit_err": err})
    R["ON"] = out


def sec_SM():
    m = rg.example_SM()
    P, err = rg.model_beta_polynomials(m, loops={"gauge": 2, "yukawa": 2, "quartic": 2}, split_loops=True)
    names = m.couplings  # g1 g2 g3 yt lam
    E = lambda **k: tuple(k.get(n, 0) for n in names)
    ref = {
        ("g1", 1): {E(g1=3): 41 / 10}, ("g2", 1): {E(g2=3): -19 / 6}, ("g3", 1): {E(g3=3): -7},
        ("g1", 2): {E(g1=5): 199 / 50, E(g1=3, g2=2): 27 / 10, E(g1=3, g3=2): 44 / 5, E(g1=3, yt=2): -17 / 10},
        ("g2", 2): {E(g1=2, g2=3): 9 / 10, E(g2=5): 35 / 6, E(g2=3, g3=2): 12, E(g2=3, yt=2): -3 / 2},
        ("g3", 2): {E(g1=2, g3=3): 11 / 10, E(g2=2, g3=3): 9 / 2, E(g3=5): -26, E(g3=3, yt=2): -2},
        ("yt", 1): {E(yt=3): 9 / 2, E(g3=2, yt=1): -8, E(g2=2, yt=1): -9 / 4, E(g1=2, yt=1): -17 / 20},
        ("lam", 1): {E(lam=2): 24, E(lam=1, yt=2): 12, E(yt=4): -6, E(g1=2, lam=1): -9 / 5, E(g2=2, lam=1): -9,
                     E(g1=4): 27 / 200, E(g1=2, g2=2): 9 / 20, E(g2=4): 9 / 8},
    }
    # literature subset of 2-loop SM coefficients (top-only, V = lam (H^+H)^2), MV / Luo-Xiao
    ref2_partial = {("lam", 2): {E(lam=3): -312, E(lam=2, yt=2): -144, E(lam=1, yt=4): -3, E(yt=6): 30,
                                 E(g3=2, lam=1, yt=2): 80, E(g3=2, yt=4): -32},
                    ("yt", 2): {E(yt=5): -12, E(g3=4, yt=1): -108, E(g3=2, yt=3): 36, E(lam=2, yt=1): 6,
                                E(lam=1, yt=3): -12}}
    rows = []
    for (b, l), d in ref.items():
        got = {k: float(v) for k, v in P[names.index(b)][l].items()}
        rows.append({"beta": b, "loop": l, "maxdev_full": maxdev(d, got)})
    for (b, l), d in ref2_partial.items():
        got = {k: float(v) for k, v in P[names.index(b)][l].items()}
        rows.append({"beta": b, "loop": l, "maxdev_on_listed_terms": max(abs(d[k] - got.get(k, 0)) for k in d)})
    polystr = {f"{n}^({l})": rg.poly_to_str(P[i][l], names) for i, n in enumerate(names) for l in P[i]}
    # thermal mass: C = 16pi^2 (3 g2^2/16 + g'^2/16 + lam/2 + yt^2/4), g'^2 = 3/5 g1^2
    rng = np.random.default_rng(0)
    tdev = 0
    for _ in range(4):
        c = rng.uniform(0, 1, 5)
        C = m.thermal_C(c)
        g1, g2, g3, yt, lam = c
        ref_m = 16 * np.pi ** 2 * (3 * g2 ** 2 / 16 + 3 / 5 * g1 ** 2 / 16 + lam / 2 + yt ** 2 / 4)
        tdev = max(tdev, np.abs(C - ref_m * np.eye(4)).max() / ref_m)
    R["SM"] = {"rows": rows, "fit_err": err, "polys": polystr, "thermal_maxreldev": tdev}


def sec_QCD():
    out = []
    for N in (2, 3, 4, 5):
        for Nf in (0, 1, 3, 6):
            m = rg.Model(f"QCD{N},{Nf}")
            m.add_gauge("SU", "SU", N, coupling="g")
            if Nf:
                m.add_dirac("q", [("SU", "fund"), Nf])
            m.finalize()
            P, err = rg.model_beta_polynomials(m, loops={"gauge": 2}, split_loops=True)
            b0 = -float(P[0][1][(3,)]); b1 = -float(P[0][2][(5,)])
            CF = (N * N - 1) / (2 * N)
            out.append({"N": N, "Nf": Nf, "b0": b0, "b0_ref": 11 * N / 3 - 2 * Nf / 3,
                        "b1": b1, "b1_ref": 34 * N * N / 3 - 10 * N * Nf / 3 - 2 * CF * Nf})
    R["QCD"] = out


def ccr_B19(N, Nf):
    x = Nf / N
    return {"g": {(2, 0, 0): -((21 - 4 * x) / 6 - 22 / (3 * N)), (3, 0, 0): ((13 * x - 27) / 6 - (23 * x - 128) / (6 * N) - 68 / (3 * N ** 2))},
            "h": {(0, 1, 1): 96 / N ** 2, (0, 2, 0): 32 / N + 16, (1, 1, 0): 6 / N - 6, (2, 0, 0): 3 / 16 - 3 / (4 * N)},
            "f": {(0, 0, 2): 64 / N ** 2 + 8, (0, 1, 1): 16 / N + 32, (0, 2, 0): 24, (1, 0, 1): 6 / N - 6, (2, 0, 0): 9 / 16}}


def sec_B19():
    out = []
    for N, Nf in [(2, 1), (3, 1), (3, 4), (4, 2), (5, 3), (6, 5), (7, 2)]:
        m = rg.example_bifund_SO(N, Nf)
        P, err = rg.model_beta_polynomials(m, loops={"gauge": 2, "quartic": 1})
        T = vh.thooft(P, m.couplings, ["g"], {"h": 1 / N, "f": 1 / N ** 2}, {"g": N})
        ref = ccr_B19(N, Nf)
        dev = max(maxdev(ref[b], T[b]) for b in ref)
        rng = np.random.default_rng(N)
        tdev = 0
        for _ in range(3):
            lam, h, f = rng.uniform(0, 0.3, 3)
            c = np.array([np.sqrt(lam / N), h / N, f / N ** 2])
            C = m.thermal_C(c)
            refm = 16 * np.pi ** 2 * (2 / 3 * (1 + 1 / (2 * N)) * h + 1 / 3 * (1 + 2 / N ** 2) * f + 1 / 8 * (1 - 1 / N) * lam)
            tdev = max(tdev, np.abs(C - refm * np.eye(m.ns)).max() / refm)
        out.append({"N": N, "Nf": Nf, "ns": m.ns, "maxdev_vs_B19": dev, "thermal_2.28_maxreldev": tdev, "fit_err": err,
                    "beta_h": vh.fmt(T["h"], ["lam", "h", "f"]), "beta_f": vh.fmt(T["f"], ["lam", "h", "f"])})
    R["B19"] = out


def planar_fit(tabs, Ns, powers, holdout=1):
    """tabs: {N: {beta: {mon: coeff}}} -> {beta: {mon: (fit dict, err)}}"""
    res = {}
    for b in tabs[Ns[0]]:
        mons = set()
        for N in Ns:
            mons |= set(tabs[N][b])
        res[b] = {}
        for mon in mons:
            vals = [tabs[N][b].get(mon, 0.0) for N in Ns]
            res[b][mon] = rg.fit_laurent(Ns, vals, powers, holdout=holdout)
    return res


def sec_RDB1():
    """RDB 1-loop quartic betas at finite N1=N2=N -> planar (CCR 3.3); also compare finite-N
    sector-diagonal parts with B.19."""
    names_t = ["lam1", "lam2", "h1", "h2", "f1", "f2", "zeta"]
    Ns = [2, 3, 4, 5, 6, 7]
    tabs = {}
    for N in Ns:
        m = rg.example_RDB(N, N, 1, 1)
        P, err = rg.model_beta_polynomials(m, loops={"gauge": 1, "quartic": 1})
        sc = {"h1": 1 / N, "h2": 1 / N, "f1": 1 / N ** 2, "f2": 1 / N ** 2, "zeta": 1 / (N * N)}
        T = vh.thooft(P, m.couplings, ["g1", "g2"], sc, {"g1": N, "g2": N})
        tabs[N] = {b: T[b] for b in ["h1", "f1", "zeta"]}
    fits = planar_fit(tabs, Ns, [0, -1, -2, -3])
    # planar reference (3.3), variables order lam1 lam2 h1 h2 f1 f2 zeta
    E = lambda **k: tuple(k.get(n, 0) for n in names_t)
    ref = {"h1": {E(h1=2): 16, E(h1=1, lam1=1): -6, E(lam1=2): 3 / 16},
           "f1": {E(f1=2): 8, E(f1=1, h1=1): 32, E(f1=1, lam1=1): -6, E(h1=2): 24, E(lam1=2): 9 / 16, E(zeta=2): 8},
           "zeta": {E(zeta=1, f1=1): 8, E(zeta=1, f2=1): 8, E(zeta=1, h1=1): 16, E(zeta=1, h2=1): 16,
                    E(zeta=1, lam1=1): -3, E(zeta=1, lam2=1): -3}}
    rows, worst, worst_fit = [], 0, 0
    for b in ref:
        planar = {mon: float(f[0][0]) for mon, f in fits[b].items()}
        planar = {k: v for k, v in planar.items() if abs(v) > 1e-9}
        worst = max(worst, maxdev(ref[b], planar))
        for mon, (fd, e) in fits[b].items():
            worst_fit = max(worst_fit, e)
            rows.append({"beta": b, "monomial": vh.fmt({mon: 1}, names_t)[3:], "laurent_in_N": {str(p): str(v) for p, v in fd.items()}, "holdout_err": e})
    R["RDB1"] = {"Ns": Ns, "planar_maxdev_vs_CCR3.3": worst, "worst_holdout": worst_fit, "coefficients": rows}


def sec_CDB1():
    names_t = ["lam1", "lam2", "h1", "h2", "f1", "f2", "zeta"]
    Ns = [2, 3, 4, 5, 6]
    tabs, gtabs = {}, {}
    for N in Ns:
        for Nf in (1, 3):
            m = rg.example_CDB(N, N, Nf, Nf)
            P, err = rg.model_beta_polynomials(m, loops={"gauge": 2, "quartic": 1})
            sc = {"h1": 1 / N, "h2": 1 / N, "f1": 1 / N ** 2, "f2": 1 / N ** 2, "zeta": 1 / (N * N)}
            T = vh.thooft(P, m.couplings, ["g1", "g2"], sc, {"g1": N, "g2": N})
            if Nf == 1:
                tabs[N] = {b: T[b] for b in ["h1", "f1", "zeta"]}
            gtabs[(N, Nf)] = T["g1"]
    fits = planar_fit(tabs, Ns, [0, -1, -2])
    E = lambda **k: tuple(k.get(n, 0) for n in names_t)
    # CCR (4.17)
    ref = {"h1": {E(h1=2): 8, E(lam1=1, h1=1): -12, E(lam1=2): 3 / 2},
           "f1": {E(f1=2): 4, E(f1=1, h1=1): 16, E(h1=2): 12, E(zeta=2): 4, E(lam1=1, f1=1): -12, E(lam1=2): 9 / 2},
           "zeta": {E(zeta=1, f1=1): 4, E(zeta=1, f2=1): 4, E(zeta=1, h1=1): 8, E(zeta=1, h2=1): 8,
                    E(zeta=1, lam1=1): -6, E(zeta=1, lam2=1): -6}}
    rows, worst, wf = [], 0, 0
    for b in ref:
        planar = {mon: float(f[0][0]) for mon, f in fits[b].items() if abs(float(f[0][0])) > 1e-9}
        worst = max(worst, maxdev(ref[b], planar))
        for mon, (fd, e) in fits[b].items():
            wf = max(wf, e)
            rows.append({"beta": b, "monomial": vh.fmt({mon: 1}, names_t)[3:], "laurent_in_N": {str(p): str(v) for p, v in fd.items()}, "holdout_err": e})
    # gauge: beta_lam1 = -(21-4x)/3 lam^2 + (-54+26x)/3 lam^3 at planar; finite-N: compare exact MV
    gdev = 0
    for (N, Nf), d in gtabs.items():
        x = Nf / N
        CF = (N * N - 1) / (2 * N)
        b0 = 11 * N / 3 - 2 * Nf / 3 - N / 6
        # 2-loop: 34/3 C_A^2 - (4 C_F + 20/3 C_A)(Nf/2) - (4 C_S + 2/3 C_A)(N/2) with C_S = C_F (complex bifund)
        b1 = 34 / 3 * N ** 2 - (4 * CF + 20 / 3 * N) * Nf / 2 - (4 * CF + 2 / 3 * N) * N / 2
        # mixed term: the other SU(N) acting on the bifundamental adds its Casimir (equal couplings)
        b1 -= 4 * CF * N / 2
        ref2 = {(2, 0, 0, 0, 0, 0, 0): -2 * b0 / N, (3, 0, 0, 0, 0, 0, 0): -2 * b1 / N ** 2}
        got = {k: v for k, v in d.items() if k[1] == 0}
        gdev = max(gdev, maxdev(ref2, got))
    R["CDB1"] = {"Ns": Ns, "planar_maxdev_vs_CCR4.17": worst, "worst_holdout": wf, "gauge_2loop_maxdev_vs_MV_formula": gdev,
                 "gauge_planar_example_N6_Nf3": vh.fmt(gtabs[(6, 3)], names_t), "coefficients": rows}


def sec_SO2L():
    """2-loop planar quartic check (CCR 3.4, h and f without zeta) from the single-sector SO(N)xSO(N) model."""
    names_t = ["lam", "h", "f"]
    Ns = [2, 3, 4, 5, 6, 7]
    tabs = {}
    for N in Ns:
        for Nf in (0, 2):
            t0 = time.time()
            m = rg.example_bifund_SO(N, Nf)
            P, err = rg.model_beta_polynomials(m, loops={"gauge": 1, "quartic": 2}, split_loops=True)
            P2 = [d.get(2, {}) if i > 0 else {} for i, d in enumerate(P)]
            T = vh.thooft(P2, m.couplings, ["g"], {"h": 1 / N, "f": 1 / N ** 2}, {"g": N})
            tabs[(N, Nf)] = {b: T[b] for b in ["h", "f"]}
            print("SO2L", N, Nf, err, round(time.time() - t0, 1), flush=True)
            R.setdefault("SO2L_raw", {})[f"{N},{Nf}"] = {b: vh.fmt(T[b], names_t) for b in ["h", "f"]}
            save("SO2L")
    # separate Nf dependence: coeff = A(N) + Nf B(N); in 't Hooft form Nf B = x_f (N B)
    A = {N: {b: tabs[(N, 0)][b] for b in ["h", "f"]} for N in Ns}
    B = {}
    for N in Ns:
        B[N] = {}
        for b in ["h", "f"]:
            keys = set(tabs[(N, 2)][b]) | set(tabs[(N, 0)][b])
            B[N][b] = {k: N * (tabs[(N, 2)][b].get(k, 0) - tabs[(N, 0)][b].get(k, 0)) / 2 for k in keys}
    fa = planar_fit(A, Ns, [0, -1, -2, -3, -4])
    fb = planar_fit(B, Ns, [0, -1, -2, -3, -4])
    E = lambda **k: tuple(k.get(n, 0) for n in names_t)
    refA = {"h": {E(h=3): -96, E(lam=1, h=2): 40, E(lam=2, h=1): -13 / 2, E(lam=3): 1 / 4},
            "f": {E(h=3): -384, E(h=2, f=1): -160, E(lam=1, h=2): 96, E(lam=1, h=1, f=1): 128, E(lam=1, f=2): 32,
                  E(lam=2, h=1): 9, E(lam=2, f=1): -7 / 2, E(lam=3): 3 / 4}}
    refB = {"h": {E(lam=2, h=1): 5 / 3, E(lam=3): -1 / 6}, "f": {E(lam=2, f=1): 5 / 3, E(lam=3): -1 / 2}}
    out = {}
    for tag, fits, ref in (("xf^0", fa, refA), ("xf^1", fb, refB)):
        wd, wf, rows = 0, 0, []
        for b in ref:
            planar = {mon: float(f[0][0]) for mon, f in fits[b].items() if abs(float(f[0][0])) > 1e-8}
            wd = max(wd, maxdev(ref[b], planar))
            for mon, (fd, e) in fits[b].items():
                wf = max(wf, e)
                rows.append({"beta": b, "monomial": vh.fmt({mon: 1}, names_t)[3:], "laurent": {str(p): str(v) for p, v in fd.items()}, "holdout_err": e})
        out[tag] = {"planar_maxdev_vs_CCR3.4": wd, "worst_holdout": wf, "rows": rows}
    R["SO2L"] = out


def example_RDB_sym(N, Nf):
    """Z2-symmetric slice of RDB: g1=g2=g, h1=h2=h, f1=f2=f (single couplings) and zeta."""
    m = rg.Model(f"RDBsym({N},{Nf})")
    for i in (1, 2):
        m.add_gauge(f"SO{i}a", "SO", N, coupling="g")
        m.add_gauge(f"SO{i}b", "SO", N, coupling="g")
    for i in (1, 2):
        m.add_real_scalar(f"Phi{i}", [(f"SO{i}a", "fund"), (f"SO{i}b", "fund")])
        if Nf:
            m.add_weyl(f"psi{i}", [(f"SO{i}a", "fund"), Nf])
            m.add_weyl(f"chi{i}", [(f"SO{i}b", "fund"), Nf])
    M = lambda p, i: p[f"Phi{i}"].T @ p[f"Phi{i}"]
    m.add_quartic("h", lambda p: sum(jnp.trace(M(p, i) @ M(p, i)) for i in (1, 2)))
    m.add_quartic("f", lambda p: sum(jnp.trace(M(p, i)) ** 2 for i in (1, 2)))
    m.add_quartic("zeta", lambda p: 2 * jnp.trace(M(p, 1)) * jnp.trace(M(p, 2)))
    return m.finalize()


def sec_RDB2L():
    names_t = ["lam", "h", "f", "zeta"]
    Ns = [2, 3, 4, 5]
    tabs = {}
    for N in Ns:
        for Nf in (0, 2):
            t0 = time.time()
            m = example_RDB_sym(N, Nf)
            P, err = rg.model_beta_polynomials(m, loops={"gauge": 1, "quartic": 2}, split_loops=True)
            P2 = [d.get(2, {}) if i > 0 else {} for i, d in enumerate(P)]
            T = vh.thooft(P2, m.couplings, ["g"], {"h": 1 / N, "f": 1 / N ** 2, "zeta": 1 / N ** 2}, {"g": N})
            tabs[(N, Nf)] = {b: T[b] for b in ["f", "zeta"]}
            print("RDB2L", N, Nf, err, round(time.time() - t0, 1), flush=True)
            R.setdefault("RDB2L_raw", {})[f"{N},{Nf}"] = {b: vh.fmt(T[b], names_t) for b in ["f", "zeta"]}
            save("RDB2L")
    A = {N: {b: tabs[(N, 0)][b] for b in ["f", "zeta"]} for N in Ns}
    B = {N: {b: {k: N * (tabs[(N, 2)][b].get(k, 0) - tabs[(N, 0)][b].get(k, 0)) / 2
                 for k in set(tabs[(N, 2)][b]) | set(tabs[(N, 0)][b])} for b in ["f", "zeta"]} for N in Ns}
    fa = planar_fit(A, Ns, [0, -1, -2])
    fb = planar_fit(B, Ns, [0, -1, -2])
    E = lambda **k: tuple(k.get(n, 0) for n in names_t)
    # CCR (3.4) on the symmetric slice: beta_zeta^(2) = zeta[-160 h^2 + 64 lam (f + 2h) + lam^2 (-21+10x)/6]
    refA = {"zeta": {E(zeta=1, h=2): -160, E(zeta=1, lam=1, f=1): 64, E(zeta=1, lam=1, h=1): 128, E(zeta=1, lam=2): -21 / 6},
            "f": {E(lam=1, zeta=2): 32}}
    refB = {"zeta": {E(zeta=1, lam=2): 10 / 6}}
    out = {}
    for tag, fits, ref in (("xf^0", fa, refA), ("xf^1", fb, refB)):
        wd, wf, rows = 0, 0, []
        for b in ref:
            planar = {mon: float(f[0][0]) for mon, f in fits[b].items() if abs(float(f[0][0])) > 1e-8}
            if b == "f":  # only the zeta-dependent terms are new relative to SO2L
                planar = {k: v for k, v in planar.items() if k[3] > 0}
            else:
                planar = {k: v for k, v in planar.items()}
            wd = max(wd, maxdev(ref[b], planar))
            for mon, (fd, e) in fits[b].items():
                if b == "f" and mon[3] == 0:
                    continue
                wf = max(wf, e)
                rows.append({"beta": b, "monomial": vh.fmt({mon: 1}, names_t)[3:], "laurent": {str(p): str(v) for p, v in fd.items()}, "holdout_err": e})
        out[tag] = {"planar_maxdev_vs_CCR3.4": wd, "worst_holdout": wf, "rows": rows}
    R["RDB2L"] = out


def sec_LS():
    """Litim-Sannino: exact finite-(Nc,Nf) polynomials in alpha variables; Veneziano limit vs
    1406.2337 eqs (3.15),(3.24),(3.25),(3.26 Delta beta_y^(2))."""
    names_a = ["ag", "ay", "ah", "av"]
    grid1 = [(Nc, Nf) for Nc in (2, 3, 4, 5, 6) for Nf in (1, 2, 3, 4, 5)]
    grid2 = [(Nc, Nf) for Nc in (2, 3, 4, 5) for Nf in (1, 2, 3, 4)]
    data = {}
    for loops, grid in (({"gauge": 2, "yukawa": 1, "quartic": 1}, grid1), ({"gauge": 1, "yukawa": 2, "quartic": 1}, grid2)):
        for Nc, Nf in grid:
            t0 = time.time()
            m = rg.example_LS(Nc, Nf)
            P, err = rg.model_beta_polynomials(m, loops=loops, split_loops=True)
            # hatted: g^2 = ag/Nc, y^2 = ay/Nc, u = ah/Nf, v = av/Nf^2
            key = "L1" if loops["yukawa"] == 1 else "L2"
            for l in (1, 2):
                Pl = [d.get(l, {}) for d in P]
                T = vh.thooft(Pl, m.couplings, ["g", "y"], {"u": 1 / Nf, "v": 1 / Nf ** 2}, {"g": Nc, "y": Nc})
                data[(key, l, Nc, Nf)] = T
            print("LS", key, Nc, Nf, err, round(time.time() - t0, 1), flush=True)
    # bivariate Laurent fits c(Nc,Nf) = sum a_pq Nc^p Nf^q
    def fit2(grid, key, l, beta, mon, P, Q):
        pts = [(Nc, Nf) for Nc, Nf in grid]
        v = np.array([data[(key, l, Nc, Nf)][beta].get(mon, 0.0) for Nc, Nf in pts])
        basis = [(p, q) for p in P for q in Q]
        A = np.array([[Nc ** p * Nf ** q for p, q in basis] for Nc, Nf in pts], float)
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        fr = {}
        for (p, q), c in zip(basis, coef):
            f, ok = rg.rationalize(c, 1000, 1e-7)
            if abs(c) > 1e-9:
                fr[(p, q)] = f if ok else c
        pred = A @ np.array([float(fr.get(b, 0)) for b in basis])
        return fr, float(np.abs(pred - v).max() / (np.abs(v).max() + 1e-300)), len(pts) - len(basis)
    out = {}
    def mons_of(key, l, beta, grid):
        s = set()
        for Nc, Nf in grid:
            s |= set(data[(key, l, Nc, Nf)][beta])
        return sorted(s)
    P1, Q1 = (-3, -2, -1, 0), (-1, 0, 1, 2)
    specs = [("L1", 1, b, grid1, P1, Q1) for b in ["g", "y", "u", "v"]] + [("L1", 2, "g", grid1, P1, Q1)] + \
            [("L2", 2, "y", grid2, (-2, -1, 0), (-1, 0, 1, 2))]
    for key, l, beta, grid, P, Q in specs:
        for mon in mons_of(key, l, beta, grid):
            fr, e, dof = fit2(grid, key, l, beta, mon, P, Q)
            # Veneziano: Nf = x Nc, Nc -> inf keep terms with p+q == 0 -> coefficient of x^q
            ven = {q: fr[(p, q)] for (p, q) in fr if p + q == 0}
            grow = [(p, q) for (p, q) in fr if p + q > 0]
            out[f"beta_{beta}^({l}) {vh.fmt({mon: 1}, names_a)[3:]}"] = {"fit": {f"Nc^{p} Nf^{q}": str(c) for (p, q), c in fr.items()},
                                                                        "fit_resid": e, "dof": dof,
                                                                        "veneziano_poly_in_x": {str(q): str(c) for q, c in ven.items()},
                                                                        "growing_terms": grow}
    R["LS"] = out


def sec_RESID():
    """Operator-completeness diagnostics: drop a basis element that the RG generates."""
    out = {}
    m = rg.Model("bifundSO_h_only")
    m.add_gauge("SOa", "SO", 4, coupling="g"); m.add_gauge("SOb", "SO", 4, coupling="g")
    m.add_real_scalar("Phi", [("SOa", "fund"), ("SOb", "fund")])
    m.add_quartic("h", lambda p: jnp.trace(p["Phi"].T @ p["Phi"] @ p["Phi"].T @ p["Phi"]))
    m.finalize()
    b, r = m.beta_quartic(np.array([0.3, 0.1]), 1, "functional")
    b2, r2 = m.beta_quartic(np.array([0.3, 0.1]), 1, "tensor")
    full = rg.example_bifund_SO(4, 0)
    bf, rf = full.beta_quartic(np.array([0.3, 0.1, 0.0]), 1, "functional")
    out["SO(4)^2 basis {h} only: residual functional/tensor"] = [r, r2]
    out["SO(4)^2 full basis {h,f}: residual"] = rf
    m = rg.Model("LS_u_only")
    m.add_gauge("SUc", "SU", 3, coupling="g"); m.add_dirac("q", [("SUc", "fund"), 2])
    m.add_complex_scalar("H", [2, 2])
    m.add_yukawa("y", lambda p, f: jnp.einsum("ai,ij,aj->", f["q"], p["H"].conj(), f["qc"]))
    m.add_quartic("u", lambda p: jnp.real(jnp.trace(p["H"].conj().T @ p["H"] @ p["H"].conj().T @ p["H"])))
    m.finalize()
    out["LS(3,2) basis {u} only: residual (v generated)"] = m.beta_quartic(np.array([0.3, 0.2, 0.1]), 1, "functional")[1]
    # N=4 bifundamental SU: det Phi is allowed only if U(1)_B is broken; with {h,f} basis and ldet != 0
    m4 = rg.example_bifund_SU(4, 0, include_det=True)
    bd, rd = m4.beta_quartic(np.array([0.3, 0.1, 0.05, 0.02]), 1, "functional")
    out["SU(4)^2 with det: residual with basis {h,f,det}"] = rd
    # drop det from basis while det coupling nonzero is impossible by construction; instead show that
    # a (det)^2-type operator is NOT generated at one loop only if in basis: test basis completeness
    out["SU(4)^2 with det: betas"] = bd
    R["RESID"] = out


def sec_FUNC():
    """Functional (field-space) vs explicit-tensor one-loop quartic betas; Yukawa 1-loop vs tensor port."""
    rows = []
    rng = np.random.default_rng(5)
    for mk in [lambda: rg.example_bifund_SU(3, 2, yukawa=True), lambda: rg.example_CDB(2, 3, 1, 2),
               lambda: rg.example_LS(3, 2), lambda: rg.example_RDB(3, 2, 1, 2), lambda: rg.example_bifund_SU(4, 1, include_det=True)]:
        m = mk()
        c = rng.uniform(-0.5, 0.5, len(m.couplings))
        bf, rf = m.beta_quartic(c, 1, "functional")
        bt, rt = m.beta_quartic(c, 1, "tensor")
        row = {"model": m.name, "ns": m.ns, "quartic_maxreldev": float(np.abs(bf - bt).max() / np.abs(bt).max()), "res_f": rf, "res_t": rt}
        if m.ny:
            gfac, y, lam = m.unpack(c)
            B1 = m.beta_yukawa_tensor1(gfac, m.Ytot(y))
            _, By = m.tensor_betas(c, loops_quartic=0, loops_yukawa=1)
            row["yukawa1_maxreldev"] = float(np.abs(B1 - By).max() / np.abs(By).max())
        rows.append(row)
    R["FUNC"] = rows


def sec_VAC():
    out = {}
    # boundedness: real SO(N)xSO(N) bifundamental V = h Tr(M^2) + f (TrM)^2, M = P^T P:
    # bounded iff f + h > 0 and f + h/N > 0 (Tr M^2 in [(TrM)^2/N, (TrM)^2]).
    N = 4
    m = rg.example_bifund_SO(N, 0)
    tests = []
    for h, f in [(1.0, 0.1), (1.0, -0.2), (1.0, -0.3), (-1.0, 1.1), (-1.0, 0.9), (-0.5, 0.2)]:
        V = m.V4(np.array([h, f]))
        bb = rg.bounded_below(V, m.ns, nstarts=40, seed=1)
        pred = min(f + h, f + h / N)
        tests.append({"h": h, "f": f, "min_on_sphere": bb["min_value"], "analytic_min": pred})
    out["bounded_below_SO4"] = tests
    # thermal minimisation + Step-5 test, complex SU(3)xSU(3) bifundamental: choose h>0, f<0 bounded,
    # and a negative thermal mass by hand (C_hat = -1): minimum should be full rank (Phi ~ 1).
    m = rg.example_bifund_SU(3, 0)
    lam = np.array([0.5, -0.1])  # h, f: bounded iff f + h/N > 0 ... f+h/3 = 0.0667>0
    V = m.V4(lam)
    Vt = lambda x: V(x) - 0.5 * jnp.sum(x * x)
    mins = rg.minimize_potential(Vt, m.ns, nstarts=30, seed=2)
    v = mins[0]["x"]
    inv = rg.matrix_invariants(m, v)
    QB = m.u1_generator({"Phi": 1.0})
    comp = rg.gauge_compensation(v, QB, m.gauge_generators_real())
    # analytic: Phi = phi 1, V = h N phi^4 + f N^2 phi^4 - N phi^2 (|z|^2 = phi^2 per diag entry)
    out["SU3_bifund_fullrank"] = {"Vmin": mins[0]["value"], "count": mins[0]["count"], "n_distinct": len(mins),
                                  "singular_values": inv["Phi"]["singular_values"], "rank": inv["Phi"]["rank"],
                                  "U1B_compensation_residual": comp["residual"]}
    # h<0 favours rank-1: V = -0.2 Tr M^2 + 0.5 (TrM)^2 -> rank 1 minimum; U(1)_B compensable (det = 0)
    lam = np.array([-0.2, 0.5])
    V = m.V4(lam)
    Vt = lambda x: V(x) - 0.5 * jnp.sum(x * x)
    mins = rg.minimize_potential(Vt, m.ns, nstarts=30, seed=3)
    v = mins[0]["x"]
    inv = rg.matrix_invariants(m, v)
    comp = rg.gauge_compensation(v, QB, m.gauge_generators_real())
    out["SU3_bifund_rank1"] = {"Vmin": mins[0]["value"], "singular_values": inv["Phi"]["singular_values"], "rank": inv["Phi"]["rank"],
                               "U1B_compensation_residual": comp["residual"]}
    # single SU(3) fundamental: U(1) rotation of a single fundamental is gauge-compensable
    m1 = rg.Model("fund")
    m1.add_gauge("SU", "SU", 3, coupling="g")
    m1.add_complex_scalar("q", [("SU", "fund")])
    m1.add_quartic("l", lambda p: jnp.real(jnp.vdot(p["q"], p["q"])) ** 2)
    m1.finalize()
    v = np.random.default_rng(0).normal(size=m1.ns)
    out["single_fundamental_U1_residual"] = rg.gauge_compensation(v, m1.u1_generator({"q": 1}), m1.gauge_generators_real())["residual"]
    # SU(3) with 2 fundamentals: SU(2) flavour on q: broken generators by a generic v (expected 3: flavour SU(2)xU(1) acts, gauge SU(3) absorbs ... )
    R["VAC"] = out


def sec_FP():
    out = {}
    # (i) Litim-Sannino Veneziano system (3.15): compare with (3.17)
    def b(x, eps):
        ag, ay = x
        return np.array([ag ** 2 * (4 / 3 * eps + (25 + 26 / 3 * eps) * ag - 2 * (11 / 2 + eps) ** 2 * ay),
                         ay * ((13 + 2 * eps) * ay - 6 * ag)])
    eps = 0.05
    sols = rg.find_fixed_points(lambda x: b(x, eps), 2, nstarts=200, bounds=([-0.2, -0.2], [0.2, 0.2]), seed=0,
                                keep=lambda x: np.linalg.norm(x) > 1e-6)
    ref = (26 * eps + 4 * eps ** 2) / (57 - 46 * eps - 8 * eps ** 2), 12 * eps / (57 - 46 * eps - 8 * eps ** 2)
    out["LS_3.15"] = {"solutions": [s["x"] for s in sols], "stability": [s["stability"] for s in sols], "ref_UV": ref,
                      "ref_BZ": (-4 * eps / (75 + 26 * eps), 0)}
    # (ii) finite-N engine fixed point for SU(3) QCD, Nf=16 two-loop BZ: g*^2 = -b0/b1
    m = rg.Model("QCD3_16"); m.add_gauge("SU", "SU", 3, coupling="g"); m.add_dirac("q", [("SU", "fund"), 16]); m.finalize()
    sols = rg.find_fixed_points(lambda x: m.beta_gauge(x, 2), 1, nstarts=40, bounds=([0.01], [0.5]),
                                keep=lambda x: x[0] > 1e-4)
    b0 = 11 - 2 * 16 / 3; b1 = 102 - 38 * 16 / 3
    out["QCD_BZ_Nf16"] = {"ghat_star": [s["x"][0] for s in sols], "ref": float(np.sqrt(-b0 / b1)), "theta": [s["stability"][0] for s in sols]}
    R["FP"] = out


def sec_PERF():
    rows = []
    for N in (4, 6, 8):
        t0 = time.time(); m = rg.example_CDB(N, N, 2, 2); tb = time.time() - t0
        c = np.random.default_rng(0).uniform(0, 0.2, len(m.couplings))
        m.beta_quartic(c, 1, "functional"); t0 = time.time(); b, r = m.beta_quartic(c, 1, "functional"); t1 = time.time() - t0
        t0 = time.time(); C = m.thermal_C(c); tc = time.time() - t0
        rows.append({"model": m.name, "ns": m.ns, "nf": m.nf, "build_s": tb, "beta_quartic_1L_s": t1, "residual": r, "thermal_C_s": tc})
        print(rows[-1], flush=True)
    for N in (8,):
        t0 = time.time(); m = rg.example_bifund_SU(N, 4, yukawa=True); tb = time.time() - t0
        c = np.random.default_rng(0).uniform(0, 0.2, len(m.couplings))
        m.beta(c); t0 = time.time(); m.beta(c); t1 = time.time() - t0
        rows.append({"model": m.name, "ns": m.ns, "nf": m.nf, "build_s": tb, "beta_full_1L_s": t1})
        print(rows[-1], flush=True)
    for N in (4, 5, 6):
        m = rg.example_bifund_SO(N, 1)
        c = np.array([0.1, 0.02, 0.003])
        t0 = time.time(); m.beta_quartic(c, 2, "tensor"); rows.append({"model": m.name, "ns": m.ns, "beta_quartic_2L_tensor_s": time.time() - t0})
        print(rows[-1], flush=True)
    R["PERF"] = rows


if __name__ == "__main__":
    for s in sys.argv[1:]:
        t0 = time.time()
        globals()["sec_" + s]()
        R.setdefault("_timing", {})[s] = time.time() - t0
        save(s)
        print("done", s, time.time() - t0, flush=True)
