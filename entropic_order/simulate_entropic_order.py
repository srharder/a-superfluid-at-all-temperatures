#!/usr/bin/env python3
r"""Entropic order in a classical lattice field theory.

Illustration for S. Harder and A. Sharon, "A Superfluid at All Temperatures",
arXiv:2610.10169.  The script shows the mechanism behind the paper's result in
a simple setting; it does not reproduce any computation of the paper.

A small sector, a two-component real field phi_x (an O(2), i.e. superfluid,
order parameter), is coupled to a reservoir of N real fields chi_x through a
negative cross coupling.  A condensate of phi lowers the mass of chi, so the
reservoir fluctuates more and carries more entropy.  At high enough temperature
this entropy gain outweighs the energy cost of the condensate, and phi orders
when the system is heated.

Model
-----
Classical field theory on a three-dimensional simple cubic lattice with L^3
sites and periodic boundary conditions, with Boltzmann weight exp(-H/T):

    H = sum_x [ 1/2 sum_mu (phi_{x+mu} - phi_x)^2 + m2/2 phi_x^2 + lam/4 (phi_x^2)^2
              + 1/2 sum_mu (chi_{x+mu} - chi_x)^2 + M2/2 chi_x^2 + g/4 (chi_x^2)^2
              - kappa/2 phi_x^2 chi_x^2 ],

with phi_x in R^2 and chi_x in R^N.  H is bounded below if kappa^2 < lam g.
A classical model keeps only the static (zero Matsubara frequency) modes of a
field theory at temperature T; the lattice spacing is the unit of length.

Parameters
----------
Entropic model: N = 32, lam = 0.1, g = 1/N, kappa = 0.9 sqrt(lam g) = 0.0503,
                m2 = +0.3, M2 = 0.5.
Control model:  the same with kappa = 0 and m2 = -0.3.  The reservoir decouples
                and phi is an ordinary superfluid, which melts when heated.

Mechanism
---------
In the Hartree approximation the phi mass is

    m_eff^2(T) ~ m2 + (4 lam - kappa N) T G,

where G = <x|1/(-Laplacian + mass^2)|x> is the lattice propagator at the
origin (G = 0.2527 for a massless field, smaller for the massive reservoir).
For kappa N > 4 lam (here 1.61 > 0.4) the thermal correction is negative and
grows with T, so phi is disordered at low T (m2 > 0) and orders above some T_c.
Solving the Hartree equations self-consistently (hartree_onset below) gives
T_c = 1.58; the Monte Carlo onset on a 12^3 lattice is T = 1.5.

Algorithm
---------
Checkerboard Metropolis: the two sublattices are updated in turn, with separate
Gaussian proposals for phi_x and for the whole vector chi_x at each site.  Step
sizes are tuned to about 45% acceptance during the first 60% of thermalization
and then held fixed.  Each temperature starts from the last configuration of
the previous one (annealing).  The first scan starts from the classical ground
state and goes up in T; a second scan goes back down to look for hysteresis.
Errors of means are sqrt(2 tau_int var / n), with the integrated
autocorrelation time tau_int measured at each T; the Binder cumulant uses a
jackknife over bins of about 2 tau_int.  Near the onset of order tau_int of the
order parameter reaches several hundred sweeps, so the points there are the
least precise.

Caveat
------
The paper's theories are conformal, so order at one temperature implies order
at all temperatures.  This lattice model has a lattice scale, so it can only
show the mechanism (order that appears upon heating), not order at all
temperatures.  After rescaling phi, chi by T^(1/4), the nearest-neighbour
coupling falls off as T^(-1/2) while the on-site terms stay fixed, so at very
high T the sites decouple and the order is lost again.  For these parameters
this happens far outside the scanned range: in short runs on an 8^3 lattice
the entropic model is still ordered at T = 10^3, has lost its order at
T = 10^4 and is fully disordered at T = 10^5.

Usage
-----
    python simulate_entropic_order.py              # default run (L = 12)
    python simulate_entropic_order.py --quick      # reduced run (L = 8)
    python simulate_entropic_order.py --check      # correctness checks
    python simulate_entropic_order.py --plot-only  # redraw figure from data
    python simulate_entropic_order.py --L 16       # other (even) lattice size

The default run writes entropic_order_data.npz, entropic_order_scan.csv,
entropic_order.png and entropic_order.pdf to --out (default: the directory of
this script).  The quick run writes the same files with the suffix _quick, and
a lattice size other than the default adds _L<size>.
Only numpy and matplotlib are needed; the run uses a single process.
"""

import argparse
import csv
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_RES = 32                        # number of reservoir components
LAM = 0.1                         # phi self-coupling
G_RES = 1.0 / N_RES               # chi self-coupling
KAPPA = 0.9 * np.sqrt(LAM * G_RES)  # cross coupling, enters H with a minus sign

MODELS = {
    "entropic": dict(m2=0.3, lam=LAM, M2=0.5, g=G_RES, kappa=KAPPA, N=N_RES),
    "control": dict(m2=-0.3, lam=LAM, M2=0.5, g=G_RES, kappa=0.0, N=N_RES),
}

RUNS = {
    # temperatures, sweeps for thermalization and measurement at each T
    "default": dict(L=12,
                    temps=np.round(np.r_[np.arange(0.2, 3.01, 0.2),
                                         np.arange(3.5, 6.01, 0.5)], 2),
                    n_therm=500, n_meas=2500),
    "quick": dict(L=8,
                  temps=np.array([0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0,
                                  4.0, 5.0, 6.0]),
                  n_therm=400, n_meas=1200),
}

SLICE_T = (0.6, 5.0)   # temperatures of the phase snapshots (entropic model)
TARGET_ACC = 0.45      # Metropolis acceptance aimed at during tuning


# ---------------------------------------------------------------------------
# Hartree estimate
# ---------------------------------------------------------------------------

def hartree_onset(P=MODELS["entropic"], n=64):
    """Temperature at which the self-consistent Hartree mass of phi vanishes,
        0 = m2 + 4 lam T G(0) - kappa N T G(M_chi^2),
        M_chi^2 = M2 + g (N+2) T G(M_chi^2) - 2 kappa T G(0),
    with G(m^2) = int d^3k/(2 pi)^3 1/(khat^2 + m^2) (midpoint rule, n^3 points)."""
    s = 4.0 * np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2
    khat2 = (s[:, None, None] + s[None, :, None] + s[None, None, :]).ravel()
    G = lambda m2: np.mean(1.0 / (khat2 + m2))
    G0 = G(0.0)

    def phi_mass2(T):
        Mc = P["M2"]
        for _ in range(30):
            Mc = P["M2"] + P["g"] * (P["N"] + 2) * T * G(Mc) - 2 * P["kappa"] * T * G0
        return P["m2"] + 4 * P["lam"] * T * G0 - P["kappa"] * P["N"] * T * G(Mc)

    lo, hi = 0.01, 50.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if phi_mass2(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Lattice and Monte Carlo update
# ---------------------------------------------------------------------------

class Lattice:
    """Simple cubic L^3 lattice with periodic boundaries, split by parity."""

    def __init__(self, L):
        if L % 2:
            raise ValueError("L must be even for the checkerboard update")
        self.L, self.V = L, L ** 3
        idx = np.arange(self.V).reshape(L, L, L)
        x, y, z = np.indices((L, L, L))
        parity = ((x + y + z) % 2).ravel()
        nbr = np.stack([np.roll(idx, s, axis=a).ravel()
                        for a in range(3) for s in (1, -1)], axis=1)
        self.sites = [np.flatnonzero(parity == p) for p in (0, 1)]
        self.nbrs = [np.ascontiguousarray(nbr[s]) for s in self.sites]


def neighbour_sum(field, nb):
    """Sum of a field over the six neighbours; field (V, n), nb (V/2, 6)."""
    total = np.take(field, nb[:, 0], axis=0)
    for k in range(1, 6):
        total += np.take(field, nb[:, k], axis=0)
    return total


def dot(a, b):
    return np.einsum("ij,ij->i", a, b)


class State:
    """Field configuration (flat arrays over sites) and Metropolis step sizes."""

    def __init__(self, lat, P):
        # classical ground state: uniform phi with phi^2 = -m2/lam if m2 < 0, chi = 0
        self.phi = np.zeros((lat.V, 2))
        if P["m2"] < 0:
            self.phi[:, 0] = np.sqrt(-P["m2"] / P["lam"])
        self.chi = np.zeros((lat.V, P["N"]))
        self.chi2 = np.zeros(lat.V)          # chi_x^2 at each site (cache)
        self.step = np.array([0.5, 0.1])     # proposal widths for phi, chi


def delta_phi(st, P, s, nb, new):
    """Change of H when phi at the sites s (one parity) is replaced by new.
    The local part of H is (6 + m2 - kappa chi^2)/2 phi^2 - phi.(neighbour sum)
    + lam/4 (phi^2)^2; sites of equal parity do not interact."""
    f = st.phi[s]
    a0, a1 = dot(f, f), dot(new, new)
    A = 6.0 + P["m2"] - P["kappa"] * st.chi2[s]
    return (0.5 * A * (a1 - a0) - dot(new - f, neighbour_sum(st.phi, nb))
            + 0.25 * P["lam"] * (a1 * a1 - a0 * a0))


def delta_chi(st, P, s, nb, new):
    """Same for the whole vector chi_x; also returns the new chi_x^2."""
    c, b0, b1 = st.chi[s], st.chi2[s], dot(new, new)
    p2 = dot(st.phi[s], st.phi[s])
    B = 6.0 + P["M2"] - P["kappa"] * p2
    dH = (0.5 * B * (b1 - b0) - dot(new - c, neighbour_sum(st.chi, nb))
          + 0.25 * P["g"] * (b1 * b1 - b0 * b0))
    return dH, b1


def metropolis(dH, T, rng):
    return rng.random(dH.shape[0]) < np.exp(np.minimum(0.0, -dH / T))


def sweep(st, lat, P, T, rng):
    """One sweep: phi and chi updates on each sublattice.  Returns acceptances."""
    acc = np.zeros(2)
    for s, nb in zip(lat.sites, lat.nbrs):
        new = st.phi[s] + st.step[0] * rng.standard_normal((len(s), 2))
        ok = metropolis(delta_phi(st, P, s, nb, new), T, rng)
        st.phi[s[ok]] = new[ok]
        acc[0] += ok.sum()
        new = st.chi[s] + st.step[1] * rng.standard_normal((len(s), P["N"]))
        dH, b1 = delta_chi(st, P, s, nb, new)
        ok = metropolis(dH, T, rng)
        st.chi[s[ok]] = new[ok]
        st.chi2[s[ok]] = b1[ok]
        acc[1] += ok.sum()
    return acc / lat.V


def measure(st, lat, P):
    """Mx, My (O(2) magnetization per site), <phi^2>, <chi^2>/N per site,
    and |sum_x chi_x|/V (to confirm that the reservoir itself stays disordered)."""
    M = st.phi.sum(axis=0) / lat.V
    return (M[0], M[1], np.sum(st.phi ** 2) / lat.V,
            st.chi2.sum() / (lat.V * P["N"]),
            np.linalg.norm(st.chi.sum(axis=0)) / lat.V)


def run_temperature(st, lat, P, T, n_therm, n_meas, rng):
    """Thermalize (tuning step sizes), then measure after every sweep."""
    n_tune, block, acc, nacc = int(0.6 * n_therm), 10, np.zeros(2), 0
    for i in range(n_therm):
        acc += sweep(st, lat, P, T, rng)
        nacc += 1
        if i < n_tune and nacc == block:
            st.step *= np.clip(acc / nacc / TARGET_ACC, 0.8, 1.25)
            acc[:], nacc = 0.0, 0
    series, acc = np.empty((n_meas, 5)), np.zeros(2)
    for i in range(n_meas):
        acc += sweep(st, lat, P, T, rng)
        series[i] = measure(st, lat, P)
    return series, acc / n_meas


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def tau_int(x, c=6.0):
    """Integrated autocorrelation time in sweeps (automatic window, Sokal)."""
    x = x - x.mean()
    if not np.any(x):
        return 0.5
    f = np.fft.rfft(x, 2 * len(x))
    rho = np.fft.irfft(f * np.conj(f))[:len(x)]
    rho /= rho[0]
    tau = 0.5
    for W in range(1, len(x)):
        tau += rho[W]
        if W >= c * tau:
            break
    return max(tau, 0.5)


def mean_err(x, tau=None):
    """Mean and its error sqrt(2 tau_int var / n), which accounts for the
    autocorrelation of successive sweeps."""
    tau = tau_int(x) if tau is None else tau
    return x.mean(), np.sqrt(2.0 * tau * x.var(ddof=1) / len(x))


def binder(m2, m4, tau):
    """U = 2 - <M^4>/<M^2>^2 (0 for a disordered, 1 for an ordered O(2) phase),
    with a jackknife error over bins of about 2 tau_int sweeps (at least 4 bins)."""
    n_bins = int(np.clip(len(m2) // max(1, int(np.ceil(2 * tau))), 4, 50))
    n = len(m2) // n_bins
    b2 = m2[len(m2) - n * n_bins:].reshape(n_bins, n).mean(axis=1)
    b4 = m4[len(m4) - n * n_bins:].reshape(n_bins, n).mean(axis=1)
    U = 2.0 - b4.mean() / b2.mean() ** 2
    jk = 2.0 - ((b4.sum() - b4) / (n_bins - 1)) / ((b2.sum() - b2) / (n_bins - 1)) ** 2
    return U, np.sqrt((n_bins - 1) / n_bins * np.sum((jk - jk.mean()) ** 2))


def analyse(series):
    mag = np.hypot(series[:, 0], series[:, 1])
    out = {"tau_mag": tau_int(mag), "tau_chi2": tau_int(series[:, 3])}
    out["mag"], out["mag_err"] = mean_err(mag, out["tau_mag"])
    out["binder"], out["binder_err"] = binder(mag ** 2, mag ** 4, out["tau_mag"])
    out["phi2"], out["phi2_err"] = mean_err(series[:, 2])
    out["chi2"], out["chi2_err"] = mean_err(series[:, 3], out["tau_chi2"])
    out["chimag"], _ = mean_err(series[:, 4])
    return out


# ---------------------------------------------------------------------------
# Temperature scans
# ---------------------------------------------------------------------------

def scan(name, temps, lat, run, rng, st=None, snapshots=None, log=None):
    """Annealing scan over temps; returns a dict of arrays and the final state.
    If snapshots is a dict, the phase of phi on the plane z = 0 is stored at the
    grid temperatures closest to SLICE_T."""
    log = log or (lambda msg: print(msg, flush=True))
    P = MODELS[name]
    if st is None:
        st = State(lat, P)
    snap_T = {float(temps[np.argmin(np.abs(temps - t))]) for t in SLICE_T}
    rows = []
    for T in temps:
        t0 = time.perf_counter()
        series, acc = run_temperature(st, lat, P, T, run["n_therm"],
                                      run["n_meas"], rng)
        r = analyse(series)
        r.update(T=T, acc_phi=acc[0], acc_chi=acc[1],
                 step_phi=st.step[0], step_chi=st.step[1])
        rows.append(r)
        if snapshots is not None and float(T) in snap_T:
            f = st.phi.reshape(lat.L, lat.L, lat.L, 2)[:, :, 0]
            snapshots[float(T)] = np.arctan2(f[..., 1], f[..., 0])
        log(f"  {name:8s} T={T:4.2f}  |M|={r['mag']:.4f}({r['mag_err']:.4f})  "
            f"U={r['binder']:+.3f}  <chi^2>/N={r['chi2']:.4f}  "
            f"acc=({acc[0]:.2f},{acc[1]:.2f})  tau_M={r['tau_mag']:.0f}  "
            f"[{time.perf_counter() - t0:.1f} s]")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}, st


def run_all(run, L, seed, log=None):
    log = log or (lambda msg: print(msg, flush=True))
    lat = Lattice(L)
    rng_e, rng_c = (np.random.default_rng(s)
                    for s in np.random.SeedSequence(seed).spawn(2))
    up, down = run["temps"], run["temps"][::-1]
    snaps, res = {}, {}
    log(f"L = {L}, {len(up)} temperatures, {run['n_therm']} + {run['n_meas']} "
        f"sweeps per temperature and scan, seed {seed}")
    res["entropic_up"], st = scan("entropic", up, lat, run, rng_e, snapshots=snaps, log=log)
    res["entropic_down"], _ = scan("entropic", down, lat, run, rng_e, st=st, log=log)
    res["control_up"], st = scan("control", up, lat, run, rng_c, log=log)
    res["control_down"], _ = scan("control", down, lat, run, rng_c, st=st, log=log)
    return res, snaps


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def ratio(num, den):
    """<chi^2>_entropic / <chi^2>_control at equal T, with propagated error."""
    order = np.argsort(den["T"])
    d = {k: v[order] for k, v in den.items()}
    i = np.searchsorted(d["T"], num["T"])
    R = num["chi2"] / d["chi2"][i]
    return R, R * np.hypot(num["chi2_err"] / num["chi2"],
                           d["chi2_err"][i] / d["chi2"][i])


def save_data(path_npz, path_csv, res, snaps, run, L, seed, runtime):
    out = {f"{scan_name}__{k}": v for scan_name, r in res.items()
           for k, v in r.items()}
    for direction in ("up", "down"):
        out[f"ratio_{direction}"], out[f"ratio_{direction}_err"] = ratio(
            res[f"entropic_{direction}"], res[f"control_{direction}"])
    out["slice_T"] = np.array(sorted(snaps))
    out["slice_phase"] = np.array([snaps[T] for T in sorted(snaps)])
    for name, P in MODELS.items():
        for k, v in P.items():
            out[f"param_{name}_{k}"] = v
    out.update(L=L, seed=seed, n_therm=run["n_therm"], n_meas=run["n_meas"],
               runtime_s=runtime)
    np.savez_compressed(path_npz, **out)
    # one row per (scan, T); chi2 is <chi^2>/N per site, chi2_ratio is
    # <chi^2>_entropic / <chi^2>_control at the same T and scan direction
    cols = ["T", "mag", "mag_err", "binder", "binder_err", "phi2", "phi2_err",
            "chi2", "chi2_err", "chimag", "acc_phi", "acc_chi", "tau_mag", "tau_chi2"]
    with open(path_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scan"] + cols + ["chi2_ratio", "chi2_ratio_err"])
        for scan_name, r in res.items():
            direction = scan_name.split("_")[1]
            for i in range(len(r["T"])):
                row = [scan_name] + [f"{r[c][i]:.6g}" for c in cols]
                if scan_name.startswith("entropic"):
                    row += [f"{out[f'ratio_{direction}'][i]:.6g}",
                            f"{out[f'ratio_{direction}_err'][i]:.6g}"]
                else:
                    row += ["", ""]
                w.writerow(row)


def make_figure(path_npz, path_png, path_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = np.load(path_npz)
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "savefig.dpi": 300, "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans", "axes.linewidth": 0.8,
    })
    c_ent, c_ctl = "#1b5e9e", "#9a9a9a"

    fig = plt.figure(figsize=(7.0, 5.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.08], hspace=0.42)
    top = gs[0].subgridspec(1, 2, wspace=0.32)
    bot = gs[1].subgridspec(1, 2, wspace=0.0)
    ax_a, ax_b = fig.add_subplot(top[0]), fig.add_subplot(top[1])

    # (a) order parameter
    for name, col, z in (("control", c_ctl, 2), ("entropic", c_ent, 3)):
        for direction, face in (("up", col), ("down", "white")):
            T = d[f"{name}_{direction}__T"]
            y, e = d[f"{name}_{direction}__mag"], d[f"{name}_{direction}__mag_err"]
            ax_a.errorbar(T, y, yerr=e, fmt="o", ms=4, mfc=face, mec=col, ecolor=col,
                          mew=1.0, lw=1.0, ls="-" if direction == "up" else "none",
                          color=col, zorder=z, capsize=0)
    ax_a.set_xlabel("temperature $T$")
    ax_a.set_ylabel(r"order parameter $\langle|\Sigma_x \phi_x|\rangle\,/\,V$")
    ax_a.text(0.03, 0.98, "control (reservoir decoupled):\nheating destroys order",
              color="#6e6e6e", ha="left", va="top", transform=ax_a.transAxes, fontsize=8)
    ax_a.text(0.97, 0.42, "entropic model:\nheating creates order",
              color=c_ent, ha="right", va="center", transform=ax_a.transAxes, fontsize=8)
    ax_a.margins(x=0.04, y=0.06)

    # (b) reservoir fluctuations
    for direction, face in (("up", c_ent), ("down", "white")):
        ax_b.errorbar(d["entropic_up__T"] if direction == "up" else d["entropic_down__T"],
                      d[f"ratio_{direction}"], yerr=d[f"ratio_{direction}_err"],
                      fmt="o", ms=4, mfc=face, mec=c_ent, ecolor=c_ent, mew=1.0,
                      lw=1.0, ls="-" if direction == "up" else "none", color=c_ent,
                      capsize=0, zorder=3)
    ax_b.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
    ax_b.set_xlabel("temperature $T$")
    ax_b.set_ylabel("reservoir fluctuations\n"
                    r"$\langle\chi^2\rangle_{\rm entropic}\,/\,\langle\chi^2\rangle_{\rm control}$")
    handles = [Line2D([], [], marker="o", ls="none", mfc="0.3", mec="0.3", ms=4),
               Line2D([], [], marker="o", ls="none", mfc="white", mec="0.3", ms=4)]
    ax_b.legend(handles, ["heating scan", "cooling scan"], loc="upper left",
                handletextpad=0.2)
    ax_b.margins(x=0.04, y=0.08)

    # (c) phase snapshots
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    phases, Ts = d["slice_phase"], d["slice_T"]
    labels = ["disordered", "ordered"]
    for j in range(len(Ts)):
        ax = fig.add_subplot(bot[j])
        im = ax.imshow(phases[j].T, origin="lower", cmap="twilight",
                       vmin=-np.pi, vmax=np.pi, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"entropic model, $T = {Ts[j]:g}$: {labels[j]}", loc="left")
        if j == 0:
            ax.set_ylabel("lattice slice $z = 0$")
            ax_c = ax
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
    cb = fig.colorbar(im, cax=cax, ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    cb.set_label(r"phase $\arg(\phi_1 + i\phi_2)$")

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="right", va="bottom")
    fig.savefig(path_png, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Correctness checks (--check)
# ---------------------------------------------------------------------------

def check(seed=1):
    """(1) local energy changes against the full Hamiltonian;
    (2) free fields (lam = g = kappa = 0) against exact lattice sums."""
    ok_all = True
    L, N = 6, N_RES
    lat, rng = Lattice(L), np.random.default_rng(seed)
    P = MODELS["entropic"]

    def H(phi, chi):
        f, c = phi.reshape(L, L, L, 2), chi.reshape(L, L, L, N)
        h = sum(0.5 * np.sum((np.roll(f, -1, a) - f) ** 2)
                + 0.5 * np.sum((np.roll(c, -1, a) - c) ** 2) for a in range(3))
        p2, c2 = np.sum(phi ** 2, 1), np.sum(chi ** 2, 1)
        return h + np.sum(0.5 * P["m2"] * p2 + 0.25 * P["lam"] * p2 ** 2
                          + 0.5 * P["M2"] * c2 + 0.25 * P["g"] * c2 ** 2
                          - 0.5 * P["kappa"] * p2 * c2)

    st = State(lat, P)
    st.phi = 1.5 * rng.standard_normal((lat.V, 2))
    st.chi = 0.8 * rng.standard_normal((lat.V, N))
    st.chi2 = np.sum(st.chi ** 2, 1)
    H0, errs = H(st.phi, st.chi), []
    for s, nb in zip(lat.sites, lat.nbrs):
        new_f = st.phi[s] + 0.5 * rng.standard_normal((len(s), 2))
        new_c = st.chi[s] + 0.3 * rng.standard_normal((len(s), N))
        d_f, (d_c, _) = delta_phi(st, P, s, nb, new_f), delta_chi(st, P, s, nb, new_c)
        for j in range(0, len(s), 7):          # single-site changes
            phi = st.phi.copy(); phi[s[j]] = new_f[j]
            chi = st.chi.copy(); chi[s[j]] = new_c[j]
            errs += [H(phi, st.chi) - H0 - d_f[j], H(st.phi, chi) - H0 - d_c[j]]
        phi = st.phi.copy(); phi[s] = new_f    # whole sublattice at once
        chi = st.chi.copy(); chi[s] = new_c
        errs += [H(phi, st.chi) - H0 - d_f.sum(), H(st.phi, chi) - H0 - d_c.sum()]
    ok = np.max(np.abs(errs)) < 1e-8
    ok_all &= ok
    print(f"[check] local energy changes vs full H ({len(errs)} cases): "
          f"max error {np.max(np.abs(errs)):.1e}  {'ok' if ok else 'FAILED'}")

    # free fields: <phi_x^2> = 2 T G_L(m2), <chi_x^2>/N = T G_L(M2), with the exact
    # finite-lattice propagator G_L(m2) = (1/V) sum_k 1/(khat^2 + m2)
    L, T, n_therm, n_meas = 8, 1.3, 1000, 8000
    Pf = dict(m2=0.4, lam=0.0, M2=0.8, g=0.0, kappa=0.0, N=N)
    lat = Lattice(L)
    k = 2 * np.pi * np.arange(L) / L
    s = 4 * np.sin(k / 2) ** 2
    khat2 = s[:, None, None] + s[None, :, None] + s[None, None, :]
    exact = {"phi2": 2 * T * np.mean(1 / (khat2 + Pf["m2"])),
             "chi2": T * np.mean(1 / (khat2 + Pf["M2"])),
             "M2": 2 * T / (lat.V * Pf["m2"])}
    st = State(lat, Pf)
    series, _ = run_temperature(st, lat, Pf, T, n_therm, n_meas, rng)
    mc = {"phi2": mean_err(series[:, 2]), "chi2": mean_err(series[:, 3]),
          "M2": mean_err(series[:, 0] ** 2 + series[:, 1] ** 2)}
    for key in ("phi2", "chi2", "M2"):
        m, e = mc[key]
        z = (m - exact[key]) / e
        ok = abs(z) < 4
        ok_all &= ok
        print(f"[check] free field {key:5s}: MC {m:.5f} +- {e:.5f}, exact "
              f"{exact[key]:.5f}, deviation {z:+.1f} sigma  {'ok' if ok else 'FAILED'}")
    print("[check] all passed" if ok_all else "[check] SOME CHECKS FAILED")
    return ok_all


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="reduced run (L = 8, fewer temperatures and sweeps)")
    ap.add_argument("--L", type=int, default=None, help="linear lattice size (even)")
    ap.add_argument("--seed", type=int, default=20261017, help="random seed")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)),
                    help="output directory")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from an existing data file")
    ap.add_argument("--check", action="store_true", help="run correctness checks")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(0 if check() else 1)

    mode = "quick" if args.quick else "default"
    run = dict(RUNS[mode])
    L = args.L if args.L is not None else run["L"]
    sfx = ("_quick" if args.quick else "") + (f"_L{L}" if L != run["L"] else "")
    os.makedirs(args.out, exist_ok=True)
    paths = {e: os.path.join(args.out, f"entropic_order{x}{sfx}.{e}")
             for e, x in (("npz", "_data"), ("csv", "_scan"), ("png", ""), ("pdf", ""))}

    if not args.plot_only:
        print(f"self-consistent Hartree estimate of the onset of order: "
              f"T = {hartree_onset():.2f}", flush=True)
        t0 = time.perf_counter()
        res, snaps = run_all(run, L, args.seed)
        runtime = time.perf_counter() - t0
        save_data(paths["npz"], paths["csv"], res, snaps, run, L, args.seed, runtime)
        print(f"Monte Carlo time {runtime:.0f} s; data written to {paths['npz']}")
    make_figure(paths["npz"], paths["png"], paths["pdf"])
    print(f"figure written to {paths['png']} and {paths['pdf']}")


if __name__ == "__main__":
    main()
