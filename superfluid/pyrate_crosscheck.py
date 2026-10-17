"""
pyrate_crosscheck.py -- independent check of the finite-N quartic beta functions with PyR@TE 3.

PyR@TE 3 (L. Sartore and I. Schienbein, arXiv:2007.12700; Apache-2.0) is downloaded at a fixed commit and run at one and
two loops on the model files in pyrate_models/: [SU(N1) x SU(N1)] x [SU(N2) x SU(N2)] with one Dirac fundamental per
SU factor and complex bifundamentals Phi_1, Phi_2, for (N1, N2) = (3,2), (3,3), (4,3), (5,2) at one loop and
(3,2), (3,3), (4,3) at two loops. Its beta functions are evaluated at random couplings, converted to the 't Hooft
couplings of the paper, and divided by
  - the closed-form one-loop quartic beta functions (Eq. (A2) of the paper), and
  - the two-loop quartic beta functions in class1_beta2L.json (Eq. (A3) of the paper).
Every ratio must equal 1.

Normalization. PyR@TE's generated RGEs.py returns the coefficient B^(L) of the L-loop term; the solver multiplies it by
1/(16 pi^2)^L. With K = 16 pi^2 and h_i = N_i h~_i / K, f_i = N_i^2 f~_i / K, zeta = N_1 N_2 z / (2K) (PyR@TE's z = 2 zeta~),
the L-loop beta function of h_i is (N_i / K) B^(L)_h~ / K^L, and similarly for f_i and zeta.

Run:  python pyrate_crosscheck.py      (needs network access to github.com, PyYAML, h5py, sympy, numpy; writes
pyrate_crosscheck.md; the two-loop PyR@TE runs take several minutes). The only change made to PyR@TE is the
EndCommands setting described below.
"""
import urllib.request, zipfile, io, os, shutil, subprocess, sys, importlib.util, inspect, json
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "pyrate_work")
COMMIT = "04b219c2016f3fc4f2371d72607edc26a7e06364"
K = 16 * np.pi ** 2
ONE_LOOP = {"CDB32": (3, 2), "CDB33": (3, 3), "CDB43": (4, 3), "CDB52": (5, 2)}
TWO_LOOP = {"CDB32": (3, 2), "CDB33": (3, 3), "CDB43": (4, 3)}
TOL = 1e-9

os.makedirs(WORK, exist_ok=True)
root = os.path.join(WORK, f"pyrate-{COMMIT}")
if not os.path.exists(root):
    data = urllib.request.urlopen(f"https://github.com/LSartore/pyrate/archive/{COMMIT}.zip", timeout=120).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(WORK)
# PyR@TE runs the commands listed under EndCommands in default.settings after every run (by default pdflatex, even
# with -no-tex). Replace them by 'cd .', which PyR@TE executes internally as a change of directory, i.e. a no-op.
settings = os.path.join(root, "default.settings")
lines = open(settings).read().splitlines()
lines = ['EndCommands : "cd ."' if l.strip().startswith("EndCommands") else l for l in lines]
open(settings, "w").write("\n".join(lines) + "\n")
for name in ONE_LOOP:
    shutil.copy(os.path.join(HERE, "pyrate_models", f"{name}.model"), os.path.join(root, "models", f"{name}.model"))


def run(name, loops):
    out = os.path.join(WORK, f"out_{loops}L")
    if not os.path.exists(os.path.join(out, name, "PythonOutput", "RGEs.py")):
        r = subprocess.run([sys.executable, "pyR@TE.py", "-m", f"models/{name}.model", "-l", str(loops),
                            "-res", out, "-no-tex", "-no-math"], cwd=root, capture_output=True, text=True, timeout=7200)
        assert r.returncode == 0, r.stderr[-2000:]
    p = os.path.join(out, name, "PythonOutput", "RGEs.py")
    spec = importlib.util.spec_from_file_location(f"rges_{name}_{loops}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {kk: getattr(m, "beta_" + kk) for kk in ["h1", "h2", "f1", "f2", "z"]}


def physical(N1, N2, l1, l2, h1, h2, f1, f2, z):
    """PyR@TE couplings from the 't Hooft couplings of the paper"""
    return dict(h1=K * h1 / N1, h2=K * h2 / N2, f1=K * f1 / N1 ** 2, f2=K * f2 / N2 ** 2, z=2 * K * z / (N1 * N2),
                g_A1=np.sqrt(K * l1 / N1), g_B1=np.sqrt(K * l1 / N1), g_A2=np.sqrt(K * l2 / N2), g_B2=np.sqrt(K * l2 / N2))


def thooft(N1, N2, B, loops):
    """L-loop 't Hooft beta functions (h1, h2, f1, f2, zeta) from PyR@TE's raw coefficients B"""
    scale = np.array([N1 / K, N2 / K, N1 ** 2 / K, N2 ** 2 / K, N1 * N2 / (2 * K)])
    return np.array([B[kk] for kk in ["h1", "h2", "f1", "f2", "z"]]) * scale / K ** loops


def evaluate(fn, loops, env):
    args = list(inspect.signature(fn).parameters)[1:]
    return float(np.real(np.asarray(fn(loops, *[env[a] for a in args])).ravel()[0]))


def closed_1loop(N1, N2, l1, l2, h1, h2, f1, f2, z):
    e1, e2, e12 = 1 / N1 ** 2, 1 / N2 ** 2, 1 / (N1 * N2)
    bh = lambda h, f, l, e: 8 * h * h - 12 * (1 - e) * l * h + (1.5 - 12 * e) * l * l + 24 * e * f * h
    bf = lambda h, f, l, e: 4 * (1 + 4 * e) * f * f + 16 * f * h - 12 * (1 - e) * l * f + 12 * h * h + (4.5 + 6 * e) * l * l + 4 * z * z
    bz = z * (4 * (1 + e1) * f1 + 4 * (1 + e2) * f2 + 8 * h1 + 8 * h2 - 6 * (1 - e1) * l1 - 6 * (1 - e2) * l2 + 8 * e12 * z)
    return np.array([bh(h1, f1, l1, e1), bh(h2, f2, l2, e2), bf(h1, f1, l1, e1), bf(h2, f2, l2, e2), bz])


J = json.load(open(os.path.join(HERE, "class1_beta2L.json")))
syms = sp.symbols("lambda1 lambda2 h1 h2 f1 f2 zeta N1 N2 x1 x2")
loc = {str(s): s for s in syms}
B2 = {kk: sp.lambdify(syms, sp.sympify(v, locals=loc), "numpy") for kk, v in J.items()}


def check(models, loops, npoints, seed):
    results = {}
    for name, (N1, N2) in models.items():
        fns = run(name, loops)
        rng = np.random.default_rng(seed)
        ratios = []
        for _ in range(npoints):
            l1, l2 = rng.uniform(0.01, 0.1, 2)
            h1, h2, f1, f2, z = rng.uniform(-0.1, 0.1, 5)
            env = physical(N1, N2, l1, l2, h1, h2, f1, f2, z)
            got = thooft(N1, N2, {kk: evaluate(fn, loops, env) for kk, fn in fns.items()}, loops)
            if loops == 1:
                ref = closed_1loop(N1, N2, l1, l2, h1, h2, f1, f2, z)
            else:   # one Dirac fundamental per SU factor: x_i = 1/N_i
                ref = np.array([B2[kk](l1, l2, h1, h2, f1, f2, z, N1, N2, 1 / N1, 1 / N2) for kk in ["h1", "h2", "f1", "f2", "z"]])
            ratios.append(got / ref)
        ratios = np.array(ratios)
        results[(N1, N2)] = (ratios.mean(0), np.abs(ratios - 1).max())
        print(f"{loops}-loop (N1,N2)={(N1, N2)}: ratios {np.round(ratios.mean(0), 12)}, max |ratio-1| {np.abs(ratios - 1).max():.1e}")
    return results


if __name__ == "__main__":
    r1 = check(ONE_LOOP, 1, 6, 1)
    r2 = check(TWO_LOOP, 2, 8, 11)
    lines = ["# PyR@TE 3 check of the finite-N quartic beta functions", "",
             f"PyR@TE 3 at commit `{COMMIT}`, run on the model files in `pyrate_models/` "
             "([SU(N1) x SU(N1)] x [SU(N2) x SU(N2)], one Dirac fundamental per SU factor, complex bifundamentals Phi_1, Phi_2). "
             "Its beta functions were evaluated at random couplings (gauge and all quartics, both signs), converted to 't Hooft "
             "couplings, and divided by the beta functions of the paper: the closed-form one-loop expressions and the two-loop "
             "expressions in `class1_beta2L.json`. Ratios are averaged over the sample points (6 at one loop, 8 at two loops).", "",
             "| loops | (N1,N2) | ratio (h1, h2, f1, f2, zeta) | max abs(ratio - 1) |", "|---|---|---|---|"]
    for loops, res in ((1, r1), (2, r2)):
        for (N1, N2), (mean, dev) in res.items():
            lines.append(f"| {loops} | {N1},{N2} | {', '.join(f'{x:.12f}' for x in mean)} | {dev:.1e} |")
    worst = max(d for res in (r1, r2) for _, d in res.values())
    lines += ["", f"The largest deviation of a ratio from 1 is {worst:.1e}."]
    open(os.path.join(HERE, "pyrate_crosscheck.md"), "w").write("\n".join(lines) + "\n")
    assert worst < TOL, f"PyR@TE disagrees with the paper's beta functions: max |ratio - 1| = {worst:.2e}"
    print(f"all ratios equal 1 to within {worst:.1e}")
