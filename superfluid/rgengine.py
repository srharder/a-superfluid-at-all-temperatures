"""
rgengine.py -- perturbative RG + thermal-potential engine for 4d renormalizable
gauge-Yukawa-scalar theories at FINITE N (explicit group matrices, no large-N input).

=====================================================================================
CONVENTIONS (read this; everything downstream depends on it)
=====================================================================================
Fields
  * Real scalars phi_a, a = 1..n_s.  A complex multiplet z (complex dim d) is stored as
    2d real components phi = sqrt(2) * [Re z, Im z]  (i.e. z = (x + i y)/sqrt 2), so that
    |D z|^2 = 1/2 (D phi)^2.
  * Two-component (left-handed) Weyl fermions psi_i, i = 1..n_f.  A Dirac fermion in R is
    two Weyl fermions, psi in R and psi^c in conj(R).  A Majorana fermion = one Weyl
    fermion in a real rep.
Lagrangian (mostly-minus metric irrelevant here)
    L = -1/4 F^2 + 1/2 (D phi)^2 + i psi^dag sigmabar.D psi
        - 1/2 ( Y^a_ij phi_a psi_i psi_j + h.c. ) - V(phi),
    V(phi) = (1/4!) lambda_abcd phi_a phi_b phi_c phi_d   (no masses: CFT targets)
Gauge group
    Product of simple factors SU(N), SO(N), Sp(N) (N = dim of fundamental, even) and U(1)s.
    Hermitian fundamental generators normalised Tr(T^A T^B) = index * delta^AB with
    index = 1/2 by DEFAULT for SU, SO and Sp (so SO(N) vector has T(vec)=1/2 and
    C2(adj)=(N-2)/2, as in Chaudhuri-Choi-Rabinovici; pass index=1 for the other common SO
    convention).  U(1): generator = charge * u1_norm.
    Scalars: real antisymmetric generators theta^A, delta phi = alpha theta^A phi.  For a
    complex multiplet with Hermitian T (delta z = i alpha T z):
        theta = [[-Im T, -Re T], [Re T, -Im T]].
    Scalar Casimir C2(S) = -sum_A theta^A theta^A (positive).  Fermions: Hermitian t^A,
    delta psi = i alpha t^A psi.
    No kinetic mixing (an error is raised if two U(1)s mix at one loop).
Scheme / normalisation of the output
    MS-bar.  ALL couplings passed to / returned from the engine are LOOP-NORMALISED:
        ghat = g/(4 pi),  yhat = y/(4 pi),  lamhat = lambda/(16 pi^2).
    Then d chat / d ln mu = beta^(1)(chat) + beta^(2)(chat) + ..., where beta^(l) is the
    usual l-loop coefficient of (16 pi^2)^-l.  E.g. 't Hooft couplings:
    g^2 N/(16 pi^2) = ghat^2 N.  beta_g is returned for g (not g^2).
Beta-function formulas
    * gauge: 1- and 2-loop Machacek-Vaughn (MV) in Weyl/real form, incl. the 2-loop Yukawa
      term  -g_k^3 Tr[C2_k(F) Y^a Y^a+]/d_k  (validated on the SM, see rgengine_validation.md).
    * Yukawa 1-loop:  beta_Y^a = 1/2[Y^b Y^b+ Y^a + Y^a Y^b+ Y^b] + 2 Y^b Y^a+ Y^b
                        + Y^b ReTr(Y^b+ Y^a) - 3 [C2(F)^T Y^a + Y^a C2(F)].
    * quartic 1-loop (functional, no rank-4 tensor):
        beta_V(phi) = 1/2 Tr[V''^2] + 3/2 sum_AB g_A^2 g_B^2 (phi th^A th^B phi)^2
                      - Tr[(M M^+)^2] + phi^T (Y2S - 3 C2S) grad V,
        M = Y^a phi_a, Y2S_ab = Re Tr(Y^a+ Y^b), C2S = -sum g_A^2 th^A th^A,
      projected onto the user basis {P_k} by least squares at random field points; the
      relative residual flags any operator generated outside the basis.
    * 2-loop quartic and 2-loop Yukawa ("tensor" method only; explicit lambda_abcd):
      general formulas in the Poole-Thomsen (arXiv:1906.04625) tensor basis with the
      numerical coefficients transcribed from PyR@TE 3 (Sartore-Schienbein 2007.12700,
      files src/Core/BetaFunctions/{Quartic,Yukawa}Couplings.py), evaluated here in the
      doubled Weyl space psi (+) psi^dagger.  Cross-checked against PyR@TE itself, the
      O(N) model, and CCR planar 2-loop quartics (see validation file).
Thermal mass (one loop, leading high-T):  M_T^2 = T^2 C,
    C_ab = lambda_abcc/24 + 1/4 sum_A g_A^2 (-th^A th^A)_ab + 1/12 Re Tr(Y^a+ Y^b)
    (Weyl fermions).  C is returned in RAW units (i.e. 16 pi^2 x the hatted expression);
    use C_hat = C/(16 pi^2) together with V_hat when minimising V_hat4 + 1/2 phi^T C_hat phi.
API CHEAT-SHEET
-------------------------------------------------------------------------------------
Model building (then .finalize(); gauge invariance of every quartic/Yukawa is checked):
  m = Model(name)
  m.add_gauge(name, 'SU'|'SO'|'SP'|'U1', N, coupling='g', index=0.5, u1_norm=1.0)
      factors sharing a coupling name are tied (e.g. Z2-symmetric SU(N)xSU(N) with one g)
  m.add_complex_scalar(name, axes, charges={U1name: q});  m.add_real_scalar(name, axes)
  m.add_weyl(name, axes, charges);  m.add_dirac(name, axes, charges)  # -> Weyl name, name+'c'
      axes: list of int (flavour axis) or (factor, 'fund'|'antifund'|'adj'|'sym2'|'antisym2'|
      'sym2bar'|'antisym2bar'|ndarray_of_generators).  Views: p[name] is the complex (or real)
      tensor with one index per axis ('adj','sym2',... axes are expanded to N x N matrices).
  m.add_quartic(name, P)      P(p) -> real polynomial; V_hat = sum_k lam_k P_k   (jax.numpy!)
  m.add_yukawa(name, W)       W(p, f) -> (1/2) Y^a_ij phi_a psi_i psi_j; L = -(y W + h.c.)
  Use poly_det() (not jnp.linalg.det) for determinant operators.
Betas (couplings in the order m.couplings = gauge + yukawa + quartic, LOOP-NORMALISED):
  m.beta(c, loops={'gauge':2,'yukawa':1,'quartic':1}, quartic_method='auto'|'functional'|
         'tensor', diagnostics=True) -> (beta vector, {'yukawa_residual','quartic_residual'})
  m.beta_gauge(c, loops) ; m.beta_yukawa(c, loops) -> (b, resid) ; m.beta_quartic(c, loops,
         method) -> (b, resid)      [loops=2 for yukawa/quartic requires the tensor method]
  model_beta_polynomials(m, loops, split_loops=False) -> ([{exponent tuple: Fraction}], err)
  PolyBeta(polys, m.couplings) -> callable beta(x) with .jac(x)
  fit_laurent(Ns, values, powers) -> ({power: Fraction}, holdout_err);  poly_to_str(d, names)
  coefficient_table_vs_N(build, Ns, loops); fit_coefficients_in_N(names, tab, powers)
Fixed points:
  find_fixed_points(beta, n, sampler=None, nstarts=300, bounds=(lo,hi), jac=None, keep=None)
      -> [{'x','residual','stability' (eigenvalues of d beta/d c),'jacobian'}]
Thermal / vacuum (hatted units, T = 1):
  m.thermal_C(c, raw=True) -> C  (M_T^2 = T^2 C; raw=False gives C/(16 pi^2))
  m.V4(lam) -> jax fn phi -> V_hat4;  m.thermal_potential(c) -> jax fn V_hat4 + phi.C_hat.phi/2
  bounded_below(V4, m.ns, nstarts) -> {'min_value','argmin','values'}   (unit sphere)
  minimize_potential(V, m.ns, nstarts) -> [{'value','x','count','xs'}]  (degenerate groups)
  matrix_invariants(m, phi) -> {multiplet: {'singular_values','rank','abs_det','norm'}}
  m.u1_generator({multiplet: q}); m.scalar_generator(name, T_hermitian);
  m.gauge_generators_real() -> (nG, ns, ns)
  gauge_compensation(v, Q, gauge_gens) -> {'residual', 'x'}  (residual ~ 0 => NOT broken)
  broken_generators(v, global_gens, gauge_gens) -> # of physically broken generators
Reference models: example_SM, example_ON, example_bifund_SO (CCR 2.16), example_RDB (CCR 2.31),
  example_bifund_SU (SU(N)^2 bifundamental + Nf Dirac fundamentals, optional Yukawa/det),
  example_CDB (CCR 2.36), example_LS (Litim-Sannino 1406.2337).
Example (SU(4)xSU(4), complex bifundamental, 20 Dirac fundamentals per factor):
  import numpy as np, rgengine as rg
  m = rg.example_bifund_SU(4, 20)                      # couplings ['g','h','f']
  P, err = rg.model_beta_polynomials(m, loops={'gauge': 2, 'quartic': 1})
  beta = rg.PolyBeta(P, m.couplings)                   # exact rational coefficients at N=4
  sols = rg.find_fixed_points(beta, 3, jac=beta.jac, bounds=([0,-.02,-.02],[.2,.02,.02]),
                              keep=lambda x: x[0] > 1e-3)
  for s in sols:                                      # (here: none -- quartics complex)
      x = s['x']; print(x, s['stability'])
      print(rg.bounded_below(m.V4(x[1:]), m.ns)['min_value'], np.linalg.eigvalsh(m.thermal_C(x)))
      mins = rg.minimize_potential(m.thermal_potential(x), m.ns, nstarts=32)
      v = mins[0]['x']; print(rg.matrix_invariants(m, v)['Phi'],
            rg.gauge_compensation(v, m.u1_generator({'Phi': 1}), m.gauge_generators_real()))
=====================================================================================
"""
from __future__ import annotations

import itertools
import math
import warnings
from fractions import Fraction

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt
import opt_einsum as _oe

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

LOOP = 16.0 * np.pi ** 2
__version__ = "1.0"


def _lstsq(A, b):
    """Least squares with a fallback driver (MKL gelsd occasionally fails under load)."""
    try:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return sla.lstsq(A, b, lapack_driver="gelsy")[0]


# ------------------------------------------------------------------------------------
# Group theory
# ------------------------------------------------------------------------------------


def _norm_gens(gens, index):
    out = []
    for G in gens:
        nrm = np.trace(G @ G).real
        out.append(G * np.sqrt(index / nrm))
    return np.array(out, dtype=complex)


def su_fund(N, index=0.5):
    """Hermitian generalized Gell-Mann generators of SU(N), Tr(T^A T^B) = index delta."""
    gens = []
    for j in range(N):
        for k in range(j + 1, N):
            S = np.zeros((N, N), complex)
            S[j, k] = S[k, j] = 1
            A = np.zeros((N, N), complex)
            A[j, k] = -1j
            A[k, j] = 1j
            gens += [S, A]
    for l in range(1, N):
        D = np.zeros((N, N), complex)
        D[np.arange(l), np.arange(l)] = 1
        D[l, l] = -l
        gens.append(D)
    return _norm_gens(gens, index)


def so_fund(N, index=0.5):
    """Imaginary antisymmetric Hermitian generators of SO(N) (vector rep)."""
    gens = []
    for j in range(N):
        for k in range(j + 1, N):
            A = np.zeros((N, N), complex)
            A[j, k] = -1j
            A[k, j] = 1j
            gens.append(A)
    return _norm_gens(gens, index)


def _orthonormal_hermitian_basis(mats, index, tol=1e-10):
    V = np.array([np.concatenate([M.real.ravel(), M.imag.ravel()]) for M in mats])
    U, s, Wt = np.linalg.svd(V, full_matrices=False)
    r = int(np.sum(s > tol * s.max()))
    n = mats[0].shape[0]
    out = []
    for w in Wt[:r]:
        M = w[: n * n].reshape(n, n) + 1j * w[n * n:].reshape(n, n)
        out.append(M)
    return _norm_gens(out, index)


def sp_fund(N, index=0.5):
    """Generators of Sp(N) = USp(N), N even: Hermitian X with X^T J + J X = 0."""
    assert N % 2 == 0
    n = N // 2
    J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    base = list(su_fund(N)) + [np.eye(N, dtype=complex)]
    proj = [(X + J @ X.T @ J) / 2 for X in base]
    proj = [P for P in proj if np.linalg.norm(P) > 1e-12]
    gens = _orthonormal_hermitian_basis(proj, index)
    assert len(gens) == n * (2 * n + 1), len(gens)
    for X in gens:
        assert np.allclose(X.T @ J + J @ X, 0)
    return gens


def structure_constants(gens, index):
    """f^{ABC} with [T^A,T^B] = i f^{ABC} T^C, Tr(T^A T^B) = index delta."""
    comm = np.einsum("aij,bjk->abik", gens, gens) - np.einsum("bij,ajk->abik", gens, gens)
    f = -1j * np.einsum("abik,cki->abc", comm, gens) / index
    assert np.allclose(f.imag, 0, atol=1e-10)
    return f.real


def adjoint_rep(gens, index):
    f = structure_constants(gens, index)
    return -1j * f  # (T_adj^A)_{BC} = -i f^{ABC}


def casimir(gens):
    C = np.einsum("aij,ajk->ik", gens, gens)
    return C


def _two_index_basis(N, sym=True):
    B = []
    for i in range(N):
        for j in range(i, N):
            if i == j and not sym:
                continue
            v = np.zeros((N, N), complex)
            if i == j:
                v[i, i] = 1.0
            else:
                v[i, j] = 1 / np.sqrt(2)
                v[j, i] = (1 if sym else -1) / np.sqrt(2)
            B.append(v.ravel())
    return np.array(B).T  # (N^2, d)


def two_index_rep(T, sym=True):
    """Symmetric / antisymmetric two-index rep of generators T (dG,N,N).
    Returns (gens (dG,d,d), embed (N,N,d)) with S_ij = sum_s embed[i,j,s] c_s."""
    N = T.shape[1]
    B = _two_index_basis(N, sym)
    I = np.eye(N)
    gens = np.array([B.conj().T @ (np.kron(t, I) + np.kron(I, t)) @ B for t in T])
    embed = B.reshape(N, N, -1)
    return gens, embed


class GaugeFactor:
    def __init__(self, name, kind, N=None, coupling=None, index=None, u1_norm=1.0):
        self.name, self.kind, self.N = name, kind.upper(), N
        self.coupling = coupling if coupling is not None else "g_" + name
        if self.kind == "U1":
            self.dim, self.C2G, self.index, self.u1_norm = 1, 0.0, None, u1_norm
            self.fund = None
            return
        self.index = 0.5 if index is None else index
        if self.kind == "SU":
            self.fund = su_fund(N, self.index)
        elif self.kind == "SO":
            self.fund = so_fund(N, self.index)
        elif self.kind == "SP":
            self.fund = sp_fund(N, self.index)
        else:
            raise ValueError(kind)
        self.dim = len(self.fund)
        self.adj = adjoint_rep(self.fund, self.index)
        cadj = casimir(self.adj)
        self.C2G = float(np.real(cadj[0, 0]))
        assert np.allclose(cadj, self.C2G * np.eye(self.dim), atol=1e-9)

    def rep(self, kind):
        """Return (gens, embed) of named rep on one tensor axis. embed=None means the axis is
        used as is; otherwise the axis is expanded to an (N,N) matrix via embed (N,N,d)."""
        if self.kind == "U1":
            raise ValueError("use charges= for U(1) factors")
        if isinstance(kind, np.ndarray):
            return kind, None
        k = kind.lower()
        if k in ("fund", "f"):
            return self.fund, None
        if k in ("antifund", "bar", "fbar", "af"):
            return -self.fund.conj(), None
        if k in ("adj", "adjoint"):
            emb = np.transpose(self.fund, (1, 2, 0)) / np.sqrt(self.index)
            return self.adj, emb
        if k in ("sym2", "s"):
            return two_index_rep(self.fund, True)
        if k in ("antisym2", "a"):
            return two_index_rep(self.fund, False)
        if k in ("sym2bar",):
            g, e = two_index_rep(self.fund, True)
            return -g.conj(), e
        if k in ("antisym2bar",):
            g, e = two_index_rep(self.fund, False)
            return -g.conj(), e
        raise ValueError(kind)


# ------------------------------------------------------------------------------------
# Multiplets
# ------------------------------------------------------------------------------------


class Multiplet:
    """A tensor multiplet. axes: list; each entry is an int (flavour axis of that size, no gauge
    action) or a tuple (factor_name, repkind) with repkind 'fund','antifund','adj','sym2',...
    or an explicit generator array."""

    def __init__(self, model, name, kind, axes, charges=None):
        self.name, self.kind = name, kind
        self.charges = dict(charges or {})
        self.axis_dims, self.axis_gens, self.axis_embed, self.axis_factor = [], [], [], []
        for ax in axes:
            if isinstance(ax, (int, np.integer)):
                self.axis_dims.append(int(ax))
                self.axis_gens.append(None)
                self.axis_embed.append(None)
                self.axis_factor.append(None)
            else:
                fname, rk = ax
                F = model.factor(fname)
                g, e = F.rep(rk)
                self.axis_dims.append(g.shape[1])
                self.axis_gens.append(g)
                self.axis_embed.append(e)
                self.axis_factor.append(fname)
        self.dim = int(np.prod(self.axis_dims)) if self.axis_dims else 1
        # dense generator matrices on the multiplet (complex dim for complex/weyl)
        self.gens = {}
        for F in model.factors:
            mats = None
            if F.kind == "U1":
                q = self.charges.get(F.name, 0)
                if q != 0:
                    mats = np.array([q * F.u1_norm * np.eye(self.dim, dtype=complex)])
            else:
                for ax, fn in enumerate(self.axis_factor):
                    if fn != F.name:
                        continue
                    g = self.axis_gens[ax]
                    blocks = []
                    for t in g:
                        ops = [np.eye(d) for d in self.axis_dims]
                        ops[ax] = t
                        M = ops[0]
                        for o in ops[1:]:
                            M = np.kron(M, o)
                        blocks.append(M)
                    blocks = np.array(blocks, dtype=complex)
                    mats = blocks if mats is None else mats + blocks
            if mats is not None:
                self.gens[F.name] = mats
        if kind == "real_scalar":
            for fn, g in self.gens.items():
                if not np.allclose(g.real, 0, atol=1e-12):
                    raise ValueError(f"real scalar {name}: rep of {fn} is not real in this basis")

    def real_gens(self, fname):
        """Real antisymmetric generators theta on the real components of this multiplet."""
        T = self.gens[fname]
        if self.kind == "real_scalar":
            return -T.imag
        TR, TI = T.real, T.imag
        return np.array([np.block([[-ti, -tr], [tr, -ti]]) for tr, ti in zip(TR, TI)])

    def view(self, comps, xp=jnp):
        X = xp.reshape(comps, tuple(self.axis_dims))
        pos = 0
        for ax, e in enumerate(self.axis_embed):
            if e is None:
                pos += 1
                continue
            X = xp.tensordot(X, e, axes=([pos], [2]))
            X = xp.moveaxis(X, (-2, -1), (pos, pos + 1))
            pos += 2
        return X


# ------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------


class Model:
    """Build a model: add_gauge, add_*_scalar, add_weyl/add_dirac, add_yukawa, add_quartic,
    then finalize().  See module docstring for conventions."""

    def __init__(self, name=""):
        self.name = name
        self.factors, self.scalars, self.fermions = [], [], []
        self.yukawas, self.quartics = [], []
        self._final = False

    # ---- construction -------------------------------------------------------------
    def factor(self, name):
        for F in self.factors:
            if F.name == name:
                return F
        raise KeyError(name)

    def add_gauge(self, name, kind, N=None, coupling=None, index=None, u1_norm=1.0):
        self.factors.append(GaugeFactor(name, kind, N, coupling, index, u1_norm))
        return self

    def add_complex_scalar(self, name, axes, charges=None):
        self.scalars.append(Multiplet(self, name, "complex_scalar", axes, charges))
        return self

    def add_real_scalar(self, name, axes):
        self.scalars.append(Multiplet(self, name, "real_scalar", axes, None))
        return self

    def add_weyl(self, name, axes, charges=None):
        self.fermions.append(Multiplet(self, name, "weyl", axes, charges))
        return self

    def add_dirac(self, name, axes, charges=None):
        """Dirac fermion = Weyl `name` in R plus Weyl `name+'c'` in conj(R)."""
        self.add_weyl(name, axes, charges)
        ch = {k: -v for k, v in (charges or {}).items()}
        self.fermions.append(_conj_multiplet(self, name + "c", axes, ch))
        return self

    def add_yukawa(self, name, W):
        """W(phi_views: dict, psi_views: dict) -> complex scalar equal to
        (1/2) Y^a_ij phi_a psi_i psi_j for unit coupling (L contains -(y W + h.c.))."""
        self.yukawas.append((name, W))
        return self

    def add_quartic(self, name, P):
        """P(phi_views: dict) -> real quartic polynomial (V contains lam * P)."""
        self.quartics.append((name, P))
        return self

    # ---- finalize ------------------------------------------------------------------
    def finalize(self, check=True, dense_limit=6e7):
        self.gauge_couplings = []
        for F in self.factors:
            if F.coupling not in self.gauge_couplings:
                self.gauge_couplings.append(F.coupling)
        self.yuk_names = [n for n, _ in self.yukawas]
        self.quart_names = [n for n, _ in self.quartics]
        self.couplings = self.gauge_couplings + self.yuk_names + self.quart_names
        self.ng, self.ny, self.nq = len(self.gauge_couplings), len(self.yuk_names), len(self.quart_names)
        # generator index ranges
        self.gen_slices, off = {}, 0
        for F in self.factors:
            self.gen_slices[F.name] = slice(off, off + F.dim)
            off += F.dim
        self.nG = off
        # scalar offsets
        off = 0
        self.s_off = {}
        for S in self.scalars:
            D = S.dim * (2 if S.kind == "complex_scalar" else 1)
            self.s_off[S.name] = (off, D)
            off += D
        self.ns = off
        off = 0
        self.f_off = {}
        for Fm in self.fermions:
            self.f_off[Fm.name] = (off, Fm.dim)
            off += Fm.dim
        self.nf = off
        # real scalar generator blocks per multiplet/factor
        self._theta_blocks = []  # (gen slice, scalar offset, D, theta(dG,D,D))
        for S in self.scalars:
            o, D = self.s_off[S.name]
            for fn in S.gens:
                self._theta_blocks.append((self.gen_slices[fn], o, D, S.real_gens(fn)))
        self._t_blocks = []
        for Fm in self.fermions:
            o, D = self.f_off[Fm.name]
            for fn, g in Fm.gens.items():
                self._t_blocks.append((self.gen_slices[fn], o, D, g))
        # per-factor Casimir matrices on scalar / fermion space (block diagonal)
        self.Cs = {F.name: np.zeros((self.ns, self.ns)) for F in self.factors}
        for sl, o, D, th in self._theta_blocks:
            fn = [F.name for F in self.factors if self.gen_slices[F.name] == sl][0]
            self.Cs[fn][o:o + D, o:o + D] += -np.einsum("aij,ajk->ik", th, th)
        self.Cf = {F.name: np.zeros((self.nf, self.nf), complex) for F in self.factors}
        for sl, o, D, t in self._t_blocks:
            fn = [F.name for F in self.factors if self.gen_slices[F.name] == sl][0]
            self.Cf[fn][o:o + D, o:o + D] += np.einsum("aij,ajk->ik", t, t)
        # dense global generators if affordable
        self.theta = None
        if self.nG * self.ns ** 2 <= dense_limit:
            self.theta = np.zeros((self.nG, self.ns, self.ns))
            for sl, o, D, th in self._theta_blocks:
                self.theta[sl, o:o + D, o:o + D] += th
        self.tgen = None
        if self.nG * self.nf ** 2 <= dense_limit:
            self.tgen = np.zeros((self.nG, self.nf, self.nf), complex)
            for sl, o, D, t in self._t_blocks:
                self.tgen[sl, o:o + D, o:o + D] += t
        # group invariants
        self._group_invariants()
        # Yukawa structures
        self.Ystruct = np.zeros((self.ny, self.ns, self.nf, self.nf), complex)
        for k, (nm, W) in enumerate(self.yukawas):
            self.Ystruct[k] = self._yukawa_tensor(W)
        # quartic functions
        self._Pfuns = []
        self._P = [jax.jit(self._wrap_P(P)) for _, P in self.quartics]
        self._Pvec = jax.jit(lambda x: jnp.stack([p(x) for p in self._Pfuns])) if self.nq else None
        self._lam_struct = None
        self._final = True
        if check:
            self.check_invariance()
        return self

    def _wrap_P(self, P):
        def f(x):
            return jnp.real(P(self.phi_views(x)))
        self._Pfuns.append(f)
        return f

    # ---- views ---------------------------------------------------------------------
    def phi_views(self, phi, xp=jnp):
        out = {}
        for S in self.scalars:
            o, D = self.s_off[S.name]
            if S.kind == "complex_scalar":
                d = S.dim
                z = (phi[o:o + d] + 1j * phi[o + d:o + 2 * d]) / np.sqrt(2)
            else:
                z = phi[o:o + D]
            out[S.name] = S.view(z, xp)
        return out

    def psi_views(self, psi, xp=jnp):
        out = {}
        for Fm in self.fermions:
            o, D = self.f_off[Fm.name]
            out[Fm.name] = Fm.view(psi[o:o + D], xp)
        return out

    def phi_from_views(self, views):
        """Inverse of phi_views for multiplets WITHOUT embedded axes (plain tensors)."""
        phi = np.zeros(self.ns)
        for S in self.scalars:
            o, D = self.s_off[S.name]
            z = np.asarray(views[S.name]).ravel()
            if S.kind == "complex_scalar":
                phi[o:o + S.dim] = np.sqrt(2) * z.real
                phi[o + S.dim:o + 2 * S.dim] = np.sqrt(2) * z.imag
            else:
                phi[o:o + D] = z.real
        return phi

    # ---- tensors -------------------------------------------------------------------
    def _yukawa_tensor(self, W):
        def Wf(phi, psi):
            return W(self.phi_views(phi), self.psi_views(psi))
        d3 = jax.jacfwd(jax.jacfwd(jax.jacfwd(Wf, argnums=1), argnums=1), argnums=0)
        Y = np.asarray(d3(jnp.zeros(self.ns), jnp.zeros(self.nf)))  # (nf,nf,ns)
        return np.transpose(Y, (2, 0, 1))

    def lambda_structures(self):
        """Explicit rank-4 tensors d^4 P_k (cached). Only for tensor-mode (small n_s)."""
        if self._lam_struct is None:
            L = np.zeros((self.nq,) + (self.ns,) * 4)
            z = jnp.zeros(self.ns)
            for k, f in enumerate(self._Pfuns):
                L[k] = np.asarray(jax.jacfwd(jax.jacfwd(jax.hessian(f)))(z))
            self._lam_struct = L
        return self._lam_struct

    def gauge_vectors(self, phi):
        """rows u_A = theta^A phi, shape (nG, ns); jax-compatible."""
        U = jnp.zeros((self.nG, self.ns), dtype=phi.dtype)
        for sl, o, D, th in self._theta_blocks:
            U = U.at[sl, o:o + D].add(jnp.einsum("aij,j->ai", th, phi[o:o + D]))
        return U

    def _group_invariants(self):
        self.S2S, self.S2F, self.dG, self.C2Gk = {}, {}, {}, {}
        for F in self.factors:
            d = F.dim
            self.dG[F.name] = d
            self.C2Gk[F.name] = F.C2G
            self.S2S[F.name] = np.trace(self.Cs[F.name]) / d
            self.S2F[F.name] = np.trace(self.Cf[F.name]).real / d
        u1 = [F for F in self.factors if F.kind == "U1"]
        for a, b in itertools.combinations(u1, 2):
            qa = np.zeros(self.nf)
            qb = np.zeros(self.nf)
            for Fm in self.fermions:
                o, D = self.f_off[Fm.name]
                qa[o:o + D] = Fm.charges.get(a.name, 0) * a.u1_norm
                qb[o:o + D] = Fm.charges.get(b.name, 0) * b.u1_norm
            sa = np.zeros(self.ns)
            sb = np.zeros(self.ns)
            for S in self.scalars:
                o, D = self.s_off[S.name]
                sa[o:o + D] = S.charges.get(a.name, 0) * a.u1_norm
                sb[o:o + D] = S.charges.get(b.name, 0) * b.u1_norm
            if abs(qa @ qb) + abs(sa @ sb) > 1e-12:
                raise NotImplementedError("U(1) kinetic mixing generated; not supported")

    # ---- checks --------------------------------------------------------------------
    def check_invariance(self, npts=3, tol=1e-8, seed=1):
        rng = np.random.default_rng(seed)
        report = {}
        for k, f in enumerate(self._Pfuns):
            g = jax.grad(f)
            worst = 0.0
            for _ in range(npts):
                x = rng.normal(size=self.ns)
                gr = np.asarray(g(jnp.array(x)))
                U = np.asarray(self.gauge_vectors(jnp.array(x)))
                worst = max(worst, (np.abs(U @ gr).max() if U.size else 0.0) / (np.linalg.norm(gr) * np.linalg.norm(x) + 1e-300))
            report["quartic:" + self.quart_names[k]] = worst
            if worst > tol:
                raise ValueError(f"quartic {self.quart_names[k]} not gauge invariant ({worst:.2e})")
        if self.ny:
            for k in range(self.ny):
                Y = self.Ystruct[k]
                if np.abs(Y).max() == 0:
                    raise ValueError(f"Yukawa {self.yuk_names[k]} vanishes identically")
                # sum_a Y^a th^A_ab + i (t^T Y^b + Y^b t) = 0, one generator at a time (low memory)
                err = 0.0
                scale = np.abs(Y).max()
                for A_ in range(self.nG):
                    L_ = np.zeros_like(Y)
                    for sl, o, D, th in self._theta_blocks:
                        if sl.start <= A_ < sl.stop:
                            L_[o:o + D] += np.tensordot(th[A_ - sl.start].T, Y[o:o + D], axes=1)
                    for sl, o, D, t in self._t_blocks:
                        if sl.start <= A_ < sl.stop:
                            tb = t[A_ - sl.start]
                            L_[:, o:o + D, :] += 1j * np.matmul(tb.T, Y[:, o:o + D, :])
                            L_[:, :, o:o + D] += 1j * np.matmul(Y[:, :, o:o + D], tb)
                    err = max(err, np.abs(L_).max() / scale)
                report["yukawa:" + self.yuk_names[k]] = err
                if err > tol:
                    raise ValueError(f"Yukawa {self.yuk_names[k]} not gauge invariant ({err:.2e})")
        self.invariance_report = report
        return report

    def _dense_theta(self):
        th = np.zeros((self.nG, self.ns, self.ns))
        for sl, o, D, b in self._theta_blocks:
            th[sl, o:o + D, o:o + D] += b
        return th

    def _dense_t(self):
        t = np.zeros((self.nG, self.nf, self.nf), complex)
        for sl, o, D, b in self._t_blocks:
            t[sl, o:o + D, o:o + D] += b
        return t

    # ---- couplings -----------------------------------------------------------------
    def unpack(self, c):
        if isinstance(c, dict):
            c = np.array([c[n] for n in self.couplings], dtype=float)
        c = np.asarray(c, dtype=float)
        gc = dict(zip(self.gauge_couplings, c[: self.ng]))
        gfac = {F.name: gc[F.coupling] for F in self.factors}
        y = c[self.ng:self.ng + self.ny]
        lam = c[self.ng + self.ny:]
        return gfac, y, lam

    def g2_per_generator(self, gfac):
        v = np.zeros(self.nG)
        for F in self.factors:
            v[self.gen_slices[F.name]] = gfac[F.name] ** 2
        return v

    def Ytot(self, y):
        if self.ny == 0:
            return np.zeros((self.ns, self.nf, self.nf), complex)
        return np.einsum("k,kaij->aij", y, self.Ystruct)

    def V4(self, lam):
        """jax function phi -> V_hat(phi) = sum_k lam_k P_k(phi)."""
        Pf = self._Pfuns
        lam = jnp.asarray(lam)

        def V(x):
            return sum(lam[k] * Pf[k](x) for k in range(len(Pf)))
        return V

    def C2S_matrix(self, gfac):
        return sum(gfac[F.name] ** 2 * self.Cs[F.name] for F in self.factors)

    def C2F_matrix(self, gfac):
        return sum(gfac[F.name] ** 2 * self.Cf[F.name] for F in self.factors)

    @staticmethod
    def Y2S_matrix(Y):
        n = Y.shape[0]
        Yf = Y.reshape(n, -1)
        return np.real(Yf.conj() @ Yf.T)

    # ---- beta functions ------------------------------------------------------------
    def beta_gauge(self, c, loops=2):
        gfac, y, lam = self.unpack(c)
        Y = self.Ytot(y)
        Y2F = _oe.contract("aji,ajk->ik", Y.conj(), Y) if self.ny else None  # Y^a+ Y^a
        out = {}
        for F in self.factors:
            k = F.name
            g = gfac[k]
            b1 = -(11 / 3 * F.C2G - 2 / 3 * self.S2F[k] - 1 / 6 * self.S2S[k])
            beta = g ** 3 * b1
            if loops >= 2:
                d = self.dG[k]
                t2 = -34 / 3 * F.C2G ** 2 * g ** 2 + 10 / 3 * F.C2G * self.S2F[k] * g ** 2 \
                    + 1 / 3 * F.C2G * self.S2S[k] * g ** 2
                for Fl in self.factors:
                    l = Fl.name
                    Fkl = np.trace(self.Cf[k] @ self.Cf[l]).real / d
                    Skl = np.trace(self.Cs[k] @ self.Cs[l]) / d
                    t2 += 2 * gfac[l] ** 2 * (Fkl + Skl)
                if self.ny:
                    t2 -= np.trace(self.Cf[k] @ Y2F).real / d
                beta += g ** 3 * t2
            out.setdefault(F.coupling, []).append(beta)
        res = []
        for gname in self.gauge_couplings:
            vals = out[gname]
            if max(vals) - min(vals) > 1e-9 * (abs(vals[0]) + 1e-300):
                warnings.warn(f"tied coupling {gname}: factor betas differ {vals}")
            res.append(vals[0])
        return np.array(res)

    def beta_yukawa_tensor1(self, gfac, Y):
        """1-loop beta tensor for Y (ns,nf,nf), Weyl form (module docstring)."""
        ein = lambda s, *a: _oe.contract(s, *a, optimize="greedy")
        Y2S = self.Y2S_matrix(Y)
        C2F = self.C2F_matrix(gfac)
        YYd = ein("bij,bkj->ik", Y, Y.conj())  # Y^b Y^b+
        YdY = ein("bji,bjk->ik", Y.conj(), Y)  # Y^b+ Y^b
        B = 0.5 * (ein("ik,akj->aij", YYd, Y) + ein("aik,kj->aij", Y, YdY))
        B += 2 * ein("bik,alk,blj->aij", Y, Y.conj(), Y)
        B += ein("ba,bij->aij", Y2S, Y)
        B -= 3 * (ein("ki,akj->aij", C2F, Y) + ein("aik,kj->aij", Y, C2F))
        return B

    def _project_yukawa(self, B):
        if self.ny == 0:
            return np.zeros(0), 0.0
        A = self.Ystruct.reshape(self.ny, -1).T
        b = B.ravel()
        coef = _lstsq(A, b)
        r = np.linalg.norm(A @ coef - b) / (np.linalg.norm(b) + 1e-300)
        if np.abs(coef.imag).max() > 1e-9 * (np.abs(coef).max() + 1e-300):
            r = max(r, np.abs(coef.imag).max() / (np.abs(coef).max() + 1e-300))
        return coef.real, r

    def sample_points(self, npts, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(npts, self.ns))
        # sparse / low-rank-ish points break large-N concentration of invariants
        for i in range(npts):
            if i % 2 == 1:
                mask = rng.random(self.ns) < rng.uniform(0.05, 0.5)
                X[i] *= mask
            if i % 3 == 2:
                for S in self.scalars:
                    o, D = self.s_off[S.name]
                    X[i, o:o + D] *= rng.exponential()
            nrm = np.linalg.norm(X[i])
            if nrm == 0:
                X[i, rng.integers(self.ns)] = 1.0
                nrm = 1.0
            X[i] /= nrm
        return X

    def _betaV_core(self):
        """Compiled once per model: (X batch, lam, g2 per generator, Y, K) -> beta_V^(1)(X)."""
        if getattr(self, "_bV_core", None) is not None:
            return self._bV_core
        Pf = self._Pfuns
        gv = self.gauge_vectors
        has_y, nG = self.ny > 0, self.nG

        def V(x, lam):
            return sum(lam[k] * Pf[k](x) for k in range(len(Pf)))
        H = jax.hessian(V, argnums=0)
        gradV = jax.grad(V, argnums=0)

        def bV(x, lam, g2, Y, K):
            h = H(x, lam)
            t = 0.5 * jnp.sum(h * h)
            if nG:
                U = gv(x)
                Gm = U @ U.T
                t = t + 1.5 * jnp.sum((g2[:, None] * g2[None, :]) * Gm ** 2)
            if has_y:
                M = jnp.einsum("a,aij->ij", x, Y)
                MM = M @ M.conj().T
                t = t - jnp.real(jnp.trace(MM @ MM))
            return t + x @ (K @ gradV(x, lam))
        self._bV_core = jax.jit(jax.vmap(bV, in_axes=(0, None, None, None, None)))
        return self._bV_core

    def betaV_1loop(self, c, X):
        """beta^(1)_V evaluated at the rows of X (functional form, see module docstring)."""
        gfac, y, lam = self.unpack(c)
        Y = self.Ytot(y)
        K = self.Y2S_matrix(Y) - 3 * self.C2S_matrix(gfac)
        return np.asarray(self._betaV_core()(jnp.asarray(X), jnp.asarray(lam),
                                              jnp.asarray(self.g2_per_generator(gfac)),
                                              jnp.asarray(Y), jnp.asarray(K)))

    def _design(self, npts, seed):
        key = (npts, seed)
        if getattr(self, "_design_cache", None) is None:
            self._design_cache = {}
        if key not in self._design_cache:
            X = self.sample_points(npts, seed)
            A = np.asarray(jax.jit(jax.vmap(lambda x: jnp.stack([p(x) for p in self._Pfuns])))(jnp.asarray(X)))
            s = np.linalg.svd(A, compute_uv=False)
            self._design_cache[key] = (X, A, s[0] / max(s[-1], 1e-300))
        return self._design_cache[key]

    def beta_quartic(self, c, loops=1, method="auto", npts=None, seed=0):
        """Return (beta vector over quartic couplings, relative residual of the projection)."""
        if self.nq == 0:
            return np.zeros(0), 0.0
        if method == "auto":
            method = "tensor" if (loops >= 2 or self.ns <= 16) else "functional"
        if method == "functional":
            if loops >= 2:
                raise ValueError("2-loop quartic needs method='tensor'")
            npts = npts or max(4 * self.nq, self.nq + 12)
            X, A, cond = self._design(npts, seed)
            self.last_quartic_condition = cond
            b = self.betaV_1loop(c, X)
            if not (np.all(np.isfinite(b)) and np.all(np.isfinite(A))):
                raise FloatingPointError("non-finite beta_V/basis values: write invariants as explicit "
                                         "polynomials (e.g. poly_det instead of jnp.linalg.det)")
            coef = _lstsq(A, b)
            r = np.linalg.norm(A @ coef - b) / (np.linalg.norm(b) + 1e-300)
            return coef, r
        Bl, _ = self.tensor_betas(c, loops_quartic=loops, loops_yukawa=0)
        return self._project_quartic_tensor(Bl)

    def _project_quartic_tensor(self, Bl):
        L = self.lambda_structures().reshape(self.nq, -1).T
        b = Bl.ravel()
        coef = _lstsq(L, b)
        r = np.linalg.norm(L @ coef - b) / (np.linalg.norm(b) + 1e-300)
        return coef, r

    def beta_yukawa(self, c, loops=1):
        gfac, y, lam = self.unpack(c)
        if self.ny == 0:
            return np.zeros(0), 0.0
        if loops == 1:
            return self._project_yukawa(self.beta_yukawa_tensor1(gfac, self.Ytot(y)))
        _, By = self.tensor_betas(c, loops_quartic=0, loops_yukawa=loops)
        return self._project_yukawa(By)

    def beta(self, c, loops=None, quartic_method="auto", diagnostics=False):
        """Full beta vector (order self.couplings) for loop-normalised couplings.
        loops: dict {'gauge':2,'yukawa':1,'quartic':1} (default)."""
        L = {"gauge": 2, "yukawa": 1, "quartic": 1}
        if loops:
            L.update(loops)
        bg = self.beta_gauge(c, L["gauge"])
        diag = {}
        if L["yukawa"] >= 2 or L["quartic"] >= 2:
            Bl, By = self.tensor_betas(c, loops_quartic=L["quartic"], loops_yukawa=L["yukawa"])
            by, ry = self._project_yukawa(By) if self.ny else (np.zeros(0), 0.0)
            bq, rq = self._project_quartic_tensor(Bl) if self.nq else (np.zeros(0), 0.0)
        else:
            by, ry = self.beta_yukawa(c, 1)
            bq, rq = self.beta_quartic(c, 1, quartic_method)
        diag = {"yukawa_residual": ry, "quartic_residual": rq}
        out = np.concatenate([bg, by, bq])
        return (out, diag) if diagnostics else out

    # ---- tensor-mode (PyR@TE / Poole-Thomsen basis) ------------------------------------
    def tensor_betas(self, c, loops_quartic=2, loops_yukawa=2):
        """Unprojected beta tensors (beta_lambda (ns^4), beta_Y (ns,nf,nf)) up to given loops,
        computed from explicit tensors in the doubled Weyl space.  Memory ~ ns^4 and
        nG^2 ns^2; practical for ns <~ 40."""
        gfac, y, lam = self.unpack(c)
        th = self.theta if self.theta is not None else self._dense_theta()
        t = self.tgen if self.tgen is not None else self._dense_t()
        g2 = self.g2_per_generator(gfac)
        C2Gv = np.zeros(self.nG)
        for F in self.factors:
            C2Gv[self.gen_slices[F.name]] = F.C2G
        Y = self.Ytot(y)
        l = np.einsum("k,kabcd->abcd", lam, self.lambda_structures()) if self.nq else np.zeros((self.ns,) * 4)
        return _pt_tensor_betas(g2, C2Gv, th, t, Y, l, loops_quartic, loops_yukawa, self.ny > 0)

    # ---- thermal -------------------------------------------------------------------
    def _C4_basis(self):
        if getattr(self, "_C4b", None) is None:
            out = []
            for f in self._Pfuns:
                lap = lambda x, f=f: jnp.trace(jax.hessian(f)(x))
                out.append(np.asarray(jax.hessian(lap)(jnp.zeros(self.ns))) / 24.0)
            self._C4b = np.array(out) if out else np.zeros((0, self.ns, self.ns))
        return self._C4b

    def thermal_C(self, c, raw=True):
        """C with M_T^2 = T^2 C at one loop.  raw=True: physical normalisation (x 16 pi^2);
        raw=False: hatted units (use with V_hat)."""
        gfac, y, lam = self.unpack(c)
        C = np.zeros((self.ns, self.ns))
        if self.nq:
            C += np.einsum("k,kab->ab", lam, self._C4_basis())
        C += 0.25 * self.C2S_matrix(gfac)
        if self.ny:
            C += self.Y2S_matrix(self.Ytot(y)) / 12.0
        C = 0.5 * (C + C.T)
        return C * LOOP if raw else C

    def thermal_potential(self, c):
        """jax function phi -> V_hat4(phi) + 1/2 phi^T C_hat phi  (T = 1, hatted units)."""
        gfac, y, lam = self.unpack(c)
        V = self.V4(lam)
        Ch = jnp.asarray(self.thermal_C(c, raw=False))
        return lambda x: V(x) + 0.5 * x @ (Ch @ x)

    # ---- symmetry generators -------------------------------------------------------
    def u1_generator(self, charges):
        """Real antisymmetric generator on scalar space for a U(1) acting as z -> e^{i q alpha} z
        on complex multiplets: charges = {multiplet_name: q}."""
        Q = np.zeros((self.ns, self.ns))
        for S in self.scalars:
            q = charges.get(S.name, 0)
            if q == 0:
                continue
            assert S.kind == "complex_scalar"
            o, D = self.s_off[S.name]
            d = S.dim
            Q[o + d:o + 2 * d, o:o + d] = q * np.eye(d)
            Q[o:o + d, o + d:o + 2 * d] = -q * np.eye(d)
        return Q

    def scalar_generator(self, name, T):
        """Real generator on scalar space from a Hermitian matrix T acting on the (complex)
        components of multiplet `name` (flattened C-order)."""
        S = [s for s in self.scalars if s.name == name][0]
        o, D = self.s_off[S.name]
        th = np.zeros((self.ns, self.ns))
        if S.kind == "complex_scalar":
            tr, ti = T.real, T.imag
            th[o:o + D, o:o + D] = np.block([[-ti, -tr], [tr, -ti]])
        else:
            th[o:o + D, o:o + D] = -T.imag
        return th

    def gauge_generators_real(self):
        return self.theta if self.theta is not None else self._dense_theta()

    def summary(self):
        s = [f"Model {self.name}: ns={self.ns} real scalars, nf={self.nf} Weyl, nG={self.nG}"]
        for F in self.factors:
            s.append(f"  {F.kind}({F.N}) '{F.name}' coupling {F.coupling}: C2G={F.C2G:.6g} "
                     f"S2F(Weyl)={self.S2F[F.name]:.6g} S2S(real)={self.S2S[F.name]:.6g}")
        s.append("  couplings: " + ", ".join(self.couplings))
        return "\n".join(s)


def _conj_multiplet(model, name, axes, charges):
    """Weyl multiplet in the conjugate of the rep specified by axes (supports embedded reps)."""
    m = Multiplet(model, name, "weyl", axes, charges)
    for fn in list(m.gens):
        F = model.factor(fn)
        if F.kind == "U1":
            continue
        m.gens[fn] = -m.gens[fn].conj()
    for ax in range(len(m.axis_gens)):
        if m.axis_gens[ax] is not None:
            m.axis_gens[ax] = -m.axis_gens[ax].conj()
            if m.axis_embed[ax] is not None:
                m.axis_embed[ax] = m.axis_embed[ax].conj()
    # U(1) gens were built with the (already negated) charges
    return m


# ------------------------------------------------------------------------------------
# General tensor formulas (doubled Weyl space), coefficients from PyR@TE 3 source
# ------------------------------------------------------------------------------------

_DEBUG = {}
# PyR@TE's fermion generator acts on the first Yukawa index from the left, i.e. equals t^T in
# our convention (verified: with t instead of t^T the 2-loop term q2_19 is not gauge covariant).
_FERMION_T_TRANSPOSE = True
_Q1 = [36, -12, 3, 2, -12]
_Q2 = [324, -684, 646, -28, -32, 12, 60, 0, 6, -143 / 3, 11 / 3, 10 / 3, -18, 24, -18, 1 / 3,
       -6, 0, -144, 60, 10, 0, -3, 0, 24, -48, 12, 0, -2, -3, 48, 24, 24]
_Y1 = [0, -6, 2, 1, 0.5]
_Y2 = [-21 / 2, 12, 0, -3, 49 / 4, -1 / 4, -1 / 2, -97 / 3, 11 / 6, 5 / 3, 1 / 12, 12, 0, 6, -12, 10,
       6, 5 / 2, 9, -1 / 2, -7 / 2, 0, -2, 2, 0, -2, 0, -1 / 2, -2, -1 / 4, -3 / 4, -1, -3 / 4]


def _sym4(B):
    out = np.zeros_like(B)
    for p in itertools.permutations(range(4)):
        out += np.transpose(B, p)
    return out / 24.0


def _pt_tensor_betas(g2, C2Gv, theta, t, Y, lam, lq, ly, has_y):
    n = theta.shape[1] if theta.size else lam.shape[0]
    nf = t.shape[1]
    ein = lambda s, *a: _oe.contract(s, *a, optimize="greedy")
    Ts = 1j * theta
    Gm = np.diag(g2)
    if _FERMION_T_TRANSPOSE:
        t = np.transpose(t, (0, 2, 1))
    # doubled fermion space
    Z = np.zeros_like(t)
    T2 = np.concatenate([np.concatenate([t, Z], 2), np.concatenate([Z, -t.conj()], 2)], 1)
    Tt = np.concatenate([np.concatenate([-t.conj(), Z], 2), np.concatenate([Z, t], 2)], 1)
    ZY = np.zeros_like(Y)
    y = np.concatenate([np.concatenate([Y, ZY], 2), np.concatenate([ZY, Y.conj()], 2)], 1)
    yt = np.concatenate([np.concatenate([Y.conj(), ZY], 2), np.concatenate([ZY, Y], 2)], 1)
    F2 = 2 * nf
    perm = np.roll(np.arange(F2), nf)
    tilde = lambda X: X[..., perm, :][..., :, perm]

    C2S = ein("A,Aac,Acb->ab", g2, Ts, Ts)
    C2F = ein("A,Aik,Akj->ij", g2, T2, T2)
    C2Ft = tilde(C2F)
    Y2S = ein("aij,bji->ab", y, yt)
    Y2F = ein("aik,akj->ij", y, yt)
    Y2Ft = tilde(Y2F)
    S2F = ein("Aij,Bji->AB", T2, T2)
    S2S = ein("Aij,Bji->AB", Ts, Ts)
    C2Gm = np.diag(C2Gv)
    GXG = lambda X: Gm @ X @ Gm
    TT = ein("Aae,Ceb->ACab", Ts, Ts)
    # pair contraction sum G_AB G_CD X^{AC}_{..} Y^{BD}_{..}
    def pairGG(X, Yt, G1=Gm, G2=Gm):
        return ein("AB,CD,ACab,BDcd->abcd", G1, G2, X, Yt)

    def pairG(X, Yt, Gpair_left=Gm, Gpair_right=Gm):
        return pairGG(X, Yt, Gpair_left, Gpair_right)

    beta_l = np.zeros((n,) * 4, dtype=complex)
    Bq1 = 0
    beta_y = np.zeros((n, F2, F2), dtype=complex)
    Y4 = ein("aij,bjk,ckl,dli->abcd", y, yt, y, yt) if has_y else None
    if lq >= 1:
        q = [pairGG(TT, TT),
             ein("ae,ebcd->abcd", C2S, lam),
             ein("abef,efcd->abcd", lam, lam),
             ein("ae,ebcd->abcd", Y2S, lam) if has_y else 0,
             Y4 if has_y else 0]
        Bq1 = sum(cf * qq for cf, qq in zip(_Q1, q))
    if lq >= 2:
        C2SG = ein("AB,Aab,Bbc->ac", GXG(C2Gm), Ts, Ts)
        C2SS = ein("AB,Aab,Bbc->ac", GXG(S2S), Ts, Ts)
        C2SF = ein("AB,Aab,Bbc->ac", GXG(S2F), Ts, Ts)
        # ---- scalar/gauge sector in real arithmetic, diagonal G (no kinetic mixing) ----
        thr = np.asarray(theta, float)
        nGg = thr.shape[0]
        for M_ in (S2S, S2F):
            assert M_.size == 0 or np.abs(M_ - np.diag(np.diag(M_))).max() <= 1e-9 * (np.abs(M_).max() + 1), "non-diagonal S2"
        s2s, s2f = np.diag(S2S).real, np.diag(S2F).real
        lamr = np.asarray(lam, float)
        C2Sr = C2S.real
        TTr = -np.einsum("Aae,Ceb->ACab", thr, thr, optimize=True)  # = Ts^A Ts^C (real)
        TTf = TTr.reshape(nGg * nGg, n * n)
        w = np.outer(g2, g2).ravel()

        def pair(X, wv, perm=None):
            # sum_{A,C} wv_{AC} X^{AC}_{ab} TT^{AC}_{cd}
            R = (X.reshape(nGg * nGg, n * n) * wv[:, None]).T @ TTf
            return R.reshape(n, n, n, n)

        def mm(A_, B_):  # A_abef B_efcd
            return (A_.reshape(n * n, n * n) @ B_.reshape(n * n, n * n)).reshape(n, n, n, n)
        q = {}
        Sop = np.zeros((n * n, n * n))  # X -> sum_A g2_A theta^A X theta^A  (vec, row-major)
        for A_ in range(nGg):
            if g2[A_] != 0:
                Sop += g2[A_] * np.kron(thr[A_], thr[A_].T)
        W = -(TTf @ Sop.T).reshape(nGg * nGg, n, n)       # W^{CE} = sum g2 Ts^A TT^{CE} Ts^A
        q[1] = pair(W, w)
        q[2] = pair(np.einsum("ai,Xib->Xab", C2Sr, TTr.reshape(-1, n, n), optimize=True), w)
        for k_, dv in ((3, C2Gv), (4, s2s), (5, s2f)):
            q[k_] = pair(TTr, np.outer(g2, g2 * g2 * dv).ravel())
        # pair(TTr, w)[a,e,b,f] = sum_AC w TT^{AC}_ae TT^{AC}_bf
        Zaebf = pair(TTr, w)
        q[6] = mm(np.transpose(Zaebf, (0, 2, 1, 3)), lamr)
        q[7] = mm(Zaebf, lamr)
        q[9] = np.einsum("af,fbcd->abcd", C2Sr @ C2Sr, lamr, optimize=True)
        q[10] = np.einsum("ae,ebcd->abcd", C2SG.real, lamr, optimize=True)
        q[11] = np.einsum("ae,ebcd->abcd", C2SS.real, lamr, optimize=True)
        q[12] = np.einsum("ae,ebcd->abcd", C2SF.real, lamr, optimize=True)
        LL = mm(lamr, lamr)
        K13 = -np.einsum("A,Aae,Abf->abef", g2, thr, thr, optimize=True)
        q[13] = mm(K13, LL)
        Lam2 = np.einsum("fg,egcd->fecd", C2Sr, lamr, optimize=True)
        q[14] = mm(lamr, Lam2)
        q[15] = np.einsum("ae,ebcd->abcd", C2Sr, LL, optimize=True)
        lamlam2 = lamr.reshape(n, -1) @ lamr.reshape(n, -1).T
        q[16] = np.einsum("ah,hbcd->abcd", lamlam2, lamr, optimize=True)
        Lr = lamr.transpose(0, 3, 1, 2).reshape(n * n, n * n)   # [(e,c),(g,h)]
        X17 = (Lr @ Lr.T).reshape(n, n, n, n).transpose(0, 2, 1, 3)  # [e,c,f,d] -> [e,f,c,d]
        q[17] = mm(lamr, X17)
        if has_y:
            Y2SCF = ein("aij,jk,bki->ab", y, C2F, yt)
            Y2SYF = ein("aij,jk,bki->ab", y, Y2Ft, yt)
            Y4S = ein("aij,cjk,bkl,cli->ab", y, yt, y, yt)
            R = ein("Dij,Bjk,ckl,dli->DBcd", T2, T2, y, yt)
            q[19] = ein("ACab,AB,CD,DBcd->abcd", TT, Gm, Gm, R)
            q[20] = ein("ae,ACeb,AB,CD,BDcd->abcd", Y2S, TT, Gm, Gm, TT)
            q[21] = ein("ae,ebcd->abcd", Y2SCF, lam)
            q[23] = ein("abef,fg,egcd->abcd", lam, Y2S, lam)
            q[25] = ein("ae,ebcd->abcd", C2S, Y4)
            q[26] = ein("aij,bjk,ckl,dlm,mi->abcd", y, yt, y, yt, C2F)
            q[27] = ein("aebf,efcd->abcd", Y4, lam)
            q[29] = ein("ae,ebcd->abcd", Y4S, lam)
            q[30] = ein("ae,ebcd->abcd", Y2SYF, lam)
            K = ein("eij,djk,ekl->dil", yt, y, yt)  # sum_e yt^e y^d yt^e
            q[31] = ein("aij,bjk,ckl,dli->abcd", y, yt, y, K)
            P = ein("eij,cjk,dkl,elm->cdim", y, yt, y, yt)  # y^e yt^c y^d yt^e
            q[32] = ein("aij,bjk,cdki->abcd", y, yt, P)
            q[33] = ein("aij,bjk,ckl,dlm,mi->abcd", y, yt, y, yt, Y2F)
        for i, cf in enumerate(_Q2, start=1):
            if cf != 0 and i in q:
                Bq1 = Bq1 + cf * q[i]
    if has_y and ly >= 1:
        Bs = []
        c1 = [ein("ba,bij->aij", C2S, y), ein("aik,kj->aij", y, C2F),
              ein("bik,akl,blj->aij", y, yt, y), ein("aik,kj->aij", y, Y2Ft),
              ein("ba,bij->aij", Y2S, y)]
        B = sum(cf * x for cf, x in zip(_Y1, c1))
        if ly >= 2:
            C2SG = ein("AB,Aab,Bbc->ac", GXG(C2Gm), Ts, Ts)
            C2SS = ein("AB,Aab,Bbc->ac", GXG(S2S), Ts, Ts)
            C2SF = ein("AB,Aab,Bbc->ac", GXG(S2F), Ts, Ts)
            C2FG = ein("AB,Aij,Bjk->ik", GXG(C2Gm), T2, T2)
            C2FS = ein("AB,Aij,Bjk->ik", GXG(S2S), T2, T2)
            C2FF = ein("AB,Aij,Bjk->ik", GXG(S2F), T2, T2)
            Y2SCF = ein("aij,jk,bki->ab", y, C2F, yt)
            Y2SYF = ein("aij,jk,bki->ab", y, Y2Ft, yt)
            Y4S = ein("aij,cjk,bkl,cli->ab", y, yt, y, yt)
            Y2FCF = ein("aij,jk,akl->il", y, C2F, yt)
            Y2FCS = ein("aij,bjk,ab->ik", y, yt, C2S)
            Y2FYF = ein("aij,jk,akl->il", y, Y2Ft, yt)
            Y2FYFt = tilde(Y2FYF)
            Y2FYS = ein("ab,aij,bjk->ik", Y2S, y, yt)
            Y4F = ein("aij,bjk,akl,blm->im", y, yt, y, yt)
            lamlam2 = ein("bcde,acde->ab", lam, lam)
            yy = {}
            yy[1] = ein("ca,cij->aij", C2S @ C2S, y)
            yy[2] = ein("ba,bik,kj->aij", C2S, y, C2F)
            yy[3] = ein("ik,akl,lj->aij", C2Ft, y, C2F)
            yy[4] = ein("aik,kl,lj->aij", y, C2F, C2F)
            yy[5] = ein("ba,bij->aij", C2SG, y)
            yy[6] = ein("ba,bij->aij", C2SS, y)
            yy[7] = ein("ba,bij->aij", C2SF, y)
            yy[8] = ein("aik,kj->aij", y, C2FG)
            yy[9] = ein("aik,kj->aij", y, C2FS)
            yy[10] = ein("aik,kj->aij", y, C2FF)
            yy[11] = ein("ab,bij->aij", lamlam2, y)
            yy[12] = ein("bik,Akl,AB,blm,Bnj,amn->aij", y, T2, Gm, yt, T2, y)
            yy[14] = ein("bik,akl,bc,clj->aij", y, yt, C2S, y)
            yy[15] = ein("ca,bik,ckl,blj->aij", C2S, y, yt, y)
            yy[16] = ein("bik,kl,alm,bmj->aij", y, C2F, yt, y)
            yy[17] = ein("bik,akl,blm,mj->aij", y, yt, y, C2F)
            yy[18] = ein("ba,bij->aij", Y2SCF, y)
            yy[19] = ein("ik,akj->aij", Y2FCS, y)
            yy[20] = ein("ik,akj->aij", Y2FCF, y)
            yy[21] = ein("aik,kl,lj->aij", y, Y2Ft, C2F)
            yy[23] = ein("bcda,bik,ckl,dlj->aij", lam, y, yt, y)
            yy[24] = ein("bik,ckl,alm,bmn,cnj->aij", y, yt, y, yt, y)
            yy[26] = ein("bik,ckl,alm,cmn,bnj->aij", y, yt, y, yt, y)
            yy[28] = ein("ba,bij->aij", Y4S, y)
            yy[29] = ein("bik,akl,lm,bmj->aij", y, yt, Y2F, y)
            yy[30] = ein("aik,kj->aij", y, Y2FYFt)
            yy[31] = ein("ba,bij->aij", Y2SYF, y)
            yy[32] = ein("bik,bc,akl,clj->aij", y, Y2S, yt, y)
            yy[33] = ein("ik,akj->aij", Y2FYS, y)
            for i, cf in enumerate(_Y2, start=1):
                if cf != 0 and i in yy:
                    B = B + cf * yy[i]
        beta_y = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    if lq >= 1:
        beta_l = _sym4(np.real(Bq1))
    lam_out = np.real(beta_l)
    return lam_out, beta_y[:, :nf, :nf]


# ------------------------------------------------------------------------------------
# Polynomial extraction and exact N-dependence
# ------------------------------------------------------------------------------------


def monomials(nvars, weights, target):
    """All exponent tuples e with sum e_i w_i == target."""
    out = []

    def rec(i, rem, cur):
        if i == nvars:
            if rem == 0:
                out.append(tuple(cur))
            return
        w = weights[i]
        for e in range(rem // w + 1):
            rec(i + 1, rem - e * w, cur + [e])
    rec(0, target, [])
    return out


def rationalize(x, max_den=100000, tol=1e-8):
    f = Fraction(x).limit_denominator(max_den)
    ok = abs(float(f) - x) <= tol * max(1.0, abs(x))
    return f, ok


def extract_polynomial(fn, nvars, weights, target_weights, nsamples=None, seed=0, scale=0.7,
                       max_den=100000, tol=1e-8):
    """Fit fn (R^nvars -> R^nout) by polynomials: output j contains the monomials whose weight
    (sum e_i w_i) is in target_weights[j] (an int or a tuple of ints, e.g. (4, 6) for a
    1+2-loop quartic beta).  Returns (list over outputs of {exponent_tuple: Fraction|float},
    max relative error of the rationalised polynomial on fresh verification points)."""
    rng = np.random.default_rng(seed)
    tw = [tuple(w) if isinstance(w, (tuple, list)) else (w,) for w in target_weights]
    groups = {}
    for j, w in enumerate(tw):
        groups.setdefault(w, []).append(j)
    monos = {w: sum((monomials(nvars, weights, wi) for wi in w), []) for w in groups}
    nmax = max(len(v) for v in monos.values())
    ns = nsamples or int(1.6 * nmax + 8)
    X = rng.uniform(-scale, scale, size=(ns, nvars))
    F = np.array([fn(x) for x in X])
    Xv = rng.uniform(-scale, scale, size=(6, nvars))
    Fv = np.array([fn(x) for x in Xv])
    res = [None] * len(tw)
    worst = 0.0
    for w, js in groups.items():
        mons = monos[w]
        A = np.array([[np.prod(x ** np.array(m)) for m in mons] for x in X])
        coef = _lstsq(A, F[:, js])
        Av = np.array([[np.prod(x ** np.array(m)) for m in mons] for x in Xv])
        for jj, j in enumerate(js):
            d = {}
            for m, cval in zip(mons, coef[:, jj]):
                if abs(cval) < 1e-10 * (np.abs(coef[:, jj]).max() + 1e-300) or abs(cval) < 1e-13:
                    continue
                f, ok = rationalize(cval, max_den, tol)
                d[m] = f if ok else float(cval)
            cv = np.array([float(d.get(m, 0)) for m in mons])
            err = np.abs(Av @ cv - Fv[:, j]).max() / (np.abs(Fv[:, j]).max() + 1e-300)
            worst = max(worst, err)
            res[j] = d
    return res, worst


def poly_to_str(d, names):
    terms = []
    for m, c in sorted(d.items(), key=lambda kv: kv[0], reverse=True):
        mon = "*".join(f"{n}^{e}" if e > 1 else n for n, e in zip(names, m) if e > 0)
        terms.append(f"({c})*{mon}" if mon else f"({c})")
    return " + ".join(terms) if terms else "0"


def model_beta_polynomials(model, loops=None, seed=0, quartic_method="auto", split_loops=False):
    """Exact (rational) polynomial form of all betas of a finalized model at its explicit N.
    Weights: g, y -> 1, lambda -> 2 (beta_g,y^(l): weight 2l+1; beta_lambda^(l): 2l+2).
    Returns (list over model.couplings of {exponents: coeff}, worst verification error).
    split_loops=True returns instead a list of dicts {loop: {exponents: coeff}}."""
    L = {"gauge": 2, "yukawa": 1, "quartic": 1}
    if loops:
        L.update(loops)
    weights = [1] * (model.ng + model.ny) + [2] * model.nq
    tw = [tuple(2 * l + 1 for l in range(1, L["gauge"] + 1))] * model.ng \
        + [tuple(2 * l + 1 for l in range(1, L["yukawa"] + 1))] * model.ny \
        + [tuple(2 * l + 2 for l in range(1, L["quartic"] + 1))] * model.nq
    fn = lambda c: model.beta(c, loops=L, quartic_method=quartic_method)
    out, worst = extract_polynomial(fn, len(model.couplings), weights, tw, seed=seed)
    if split_loops:
        sp = []
        for d in out:
            dd = {}
            for mon, cf in d.items():
                w = sum(e * wi for e, wi in zip(mon, weights))
                l = (w - 1) // 2 if w % 2 == 1 else (w - 2) // 2
                dd.setdefault(l, {})[mon] = cf
            sp.append(dd)
        return sp, worst
    return out, worst


class PolyBeta:
    """Fast polynomial beta functions from coefficient dicts (e.g. model_beta_polynomials)."""

    def __init__(self, polys, names):
        self.names = list(names)
        self.n = len(names)
        self.terms = []
        for d in polys:
            mons = np.array(list(d.keys()), dtype=int).reshape(-1, self.n)
            cf = np.array([float(v) for v in d.values()])
            self.terms.append((mons, cf))

    def __call__(self, x):
        x = np.asarray(x, float)
        return np.array([cf @ np.prod(x[None, :] ** m, axis=1) if len(cf) else 0.0 for m, cf in self.terms])

    def jac(self, x):
        x = np.asarray(x, float)
        J = np.zeros((self.n, self.n))
        for i, (m, cf) in enumerate(self.terms):
            if not len(cf):
                continue
            for k in range(self.n):
                mk = m.copy()
                e = mk[:, k].astype(float)
                mk[:, k] = np.maximum(mk[:, k] - 1, 0)
                J[i, k] = np.sum(cf * e * np.prod(x[None, :] ** mk, axis=1))
        return J


def fit_laurent(Ns, values, powers, max_den=100000, tol=1e-7, holdout=1):
    """Fit values(N) = sum_{p in powers} a_p N^p exactly; the last `holdout` points are used only
    for verification.  Returns ({p: Fraction}, max relative holdout error)."""
    Ns = np.asarray(Ns, float)
    v = np.asarray(values, float)
    nfit = len(Ns) - holdout
    assert nfit >= len(powers), "need more N values"
    A = np.array([[N ** p for p in powers] for N in Ns])
    coef = _lstsq(A[:nfit], v[:nfit])
    fr = {}
    for p, cval in zip(powers, coef):
        f, ok = rationalize(cval, max_den, tol)
        fr[p] = f if ok else float(cval)
    pred = A @ np.array([float(fr[p]) for p in powers])
    err = np.abs(pred - v).max() / (np.abs(v).max() + 1e-300)
    return fr, err


def coefficient_table_vs_N(build_model, Ns, loops=None, seed=0):
    """Evaluate exact polynomial betas for build_model(N) at each N in Ns.
    Returns (names, {N: polys}, worst error)."""
    tab, worst, names = {}, 0.0, None
    for N in Ns:
        m = build_model(N)
        polys, err = model_beta_polynomials(m, loops, seed)
        tab[N] = polys
        names = m.couplings
        worst = max(worst, err)
    return names, tab, worst


def fit_coefficients_in_N(names, tab, powers, holdout=1):
    """For every (beta, monomial) fit exact Laurent polynomial in N. Returns dict
    {(beta_name, monomial): ({p: coeff}, err)}."""
    Ns = sorted(tab)
    keys = set()
    for N in Ns:
        for j, d in enumerate(tab[N]):
            for m in d:
                keys.add((j, m))
    out = {}
    for j, m in sorted(keys):
        vals = [float(tab[N][j].get(m, 0)) for N in Ns]
        out[(names[j], m)] = fit_laurent(Ns, vals, powers, holdout=holdout)
    return out


# ------------------------------------------------------------------------------------
# Fixed points
# ------------------------------------------------------------------------------------


def find_fixed_points(beta, n, sampler=None, nstarts=300, seed=0, jac=None, tol=1e-11,
                      dedupe=1e-7, bounds=None, keep=None):
    """All real zeros of beta: R^n -> R^n found by multistart hybrid Powell + Newton polish.
    sampler(rng) -> start point (default uniform in bounds or [-1,1]^n).
    Returns list of dicts {x, residual, stability (eigenvalues of d beta/d c), jacobian} sorted
    by |x|.  keep(x) -> bool optional filter (e.g. physical region)."""
    rng = np.random.default_rng(seed)
    if sampler is None:
        lo, hi = (np.full(n, -1.0), np.full(n, 1.0)) if bounds is None else map(np.asarray, bounds)
        sampler = lambda r: r.uniform(lo, hi)

    def J(x):
        if jac is not None:
            return jac(x)
        h = 1e-6 * (1 + np.abs(x))
        M = np.zeros((n, n))
        for k in range(n):
            e = np.zeros(n)
            e[k] = h[k]
            M[:, k] = (beta(x + e) - beta(x - e)) / (2 * h[k])
        return M

    sols = []
    for _ in range(nstarts):
        x0 = sampler(rng)
        try:
            r = sopt.root(beta, x0, jac=J, method="hybr", tol=1e-14)
        except Exception:
            continue
        x = r.x
        for _ in range(6):  # Newton polish
            try:
                dx = np.linalg.solve(J(x), -beta(x))
            except np.linalg.LinAlgError:
                break
            x = x + dx
            if np.linalg.norm(dx) < 1e-15 * (1 + np.linalg.norm(x)):
                break
        res = np.linalg.norm(beta(x))
        if not np.all(np.isfinite(x)) or res > tol * (1 + np.linalg.norm(x) ** 2):
            continue
        if keep is not None and not keep(x):
            continue
        if any(np.linalg.norm(x - s["x"]) < dedupe * (1 + np.linalg.norm(x)) for s in sols):
            continue
        Jx = J(x)
        ev = np.linalg.eigvals(Jx)
        sols.append({"x": x, "residual": res, "stability": ev[np.argsort(ev.real)], "jacobian": Jx})
    sols.sort(key=lambda s: np.linalg.norm(s["x"]))
    return sols


# ------------------------------------------------------------------------------------
# Vacuum tools
# ------------------------------------------------------------------------------------


def _multistart(fun, n, nstarts, seed, x0s=None, scale=1.0, sphere=False):
    f = jax.jit(fun)
    vg = jax.jit(jax.value_and_grad(fun))
    rng = np.random.default_rng(seed)
    out = []
    starts = list(x0s) if x0s is not None else []
    while len(starts) < nstarts:
        x = rng.normal(size=n)
        if len(starts) % 2 == 1:
            x *= rng.random(n) < rng.uniform(0.1, 0.7)
            if not x.any():
                x[rng.integers(n)] = 1
        starts.append(scale * x / np.linalg.norm(x))

    def fg(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)
    for x0 in starts:
        r = sopt.minimize(fg, x0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-12, "ftol": 1e-15})
        out.append((float(f(jnp.asarray(r.x))), r.x))
    return out


def bounded_below(V4, n, nstarts=64, seed=0, x0s=None):
    """Minimise V4 on the unit sphere of field space (multistart). Returns dict with
    min_value (>0: strictly bounded; ~0: flat direction; <0: unbounded), argmin, all values."""
    def on_sphere(x):
        return V4(x / jnp.sqrt(jnp.sum(x * x)))
    res = _multistart(on_sphere, n, nstarts, seed, x0s)
    vals = np.array([v for v, _ in res])
    i = int(np.argmin(vals))
    x = res[i][1] / np.linalg.norm(res[i][1])
    return {"min_value": vals[i], "argmin": x, "values": np.sort(vals)}


def minimize_potential(Vfun, n, nstarts=64, seed=0, x0s=None, scale=1.0, cluster_tol=1e-7):
    """Global multistart minimisation of Vfun (e.g. model.thermal_potential(c)).  Returns list
    of distinct minima (by value) sorted: dicts {value, x, count}.  Degenerate minima of equal
    value are grouped; inspect x (e.g. with matrix_invariants) to tell orientations apart."""
    res = _multistart(Vfun, n, nstarts, seed, x0s, scale)
    res.sort(key=lambda t: t[0])
    groups = []
    for v, x in res:
        for g in groups:
            if abs(v - g["value"]) <= cluster_tol * (1 + abs(v)):
                g["count"] += 1
                g["xs"].append(x)
                break
        else:
            groups.append({"value": v, "x": x, "count": 1, "xs": [x]})
    return groups


def matrix_invariants(model, phi, tol=1e-6):
    """Singular values, ranks and |det| of every matrix-valued (2-axis) scalar multiplet at phi."""
    views = model.phi_views(np.asarray(phi), xp=np)
    out = {}
    for name, X in views.items():
        X = np.asarray(X)
        entry = {"norm": float(np.linalg.norm(X))}
        if X.ndim == 2:
            s = np.linalg.svd(X, compute_uv=False)
            entry["singular_values"] = s
            entry["rank"] = int(np.sum(s > tol * max(s.max(), 1e-300)))
            if X.shape[0] == X.shape[1]:
                entry["abs_det"] = float(abs(np.linalg.det(X)))
        out[name] = entry
    return out


def gauge_compensation(v, Q, gauge_gens):
    """Step-5 test.  Solve Q v = sum_a x_a T_a v in least squares (T_a real antisymmetric gauge
    generators on real scalar space).  Returns dict(residual=|Qv - sum x T v|/|Qv|, x).
    residual ~ 0  =>  the condensate does NOT break the global symmetry generated by Q."""
    v = np.asarray(v, float)
    Qv = Q @ v
    if np.linalg.norm(Qv) < 1e-14 * (np.linalg.norm(v) + 1e-300):
        return {"residual": 0.0, "x": None, "Qv_norm": 0.0, "note": "Qv = 0: Q unbroken trivially"}
    A = np.einsum("aij,j->ia", gauge_gens, v)
    x = _lstsq(A, Qv)
    r = np.linalg.norm(A @ x - Qv) / np.linalg.norm(Qv)
    return {"residual": float(r), "x": x, "Qv_norm": float(np.linalg.norm(Qv))}


def broken_generators(v, global_gens, gauge_gens, tol=1e-8):
    """Number of global generators broken by v modulo gauge transformations:
    rank([T v, Q v]) - rank([T v])  (T: gauge, Q: global generators, all real antisymmetric)."""
    v = np.asarray(v, float)
    Tv = np.einsum("aij,j->ia", gauge_gens, v) if len(gauge_gens) else np.zeros((len(v), 0))
    Qv = np.einsum("aij,j->ia", global_gens, v)
    def rank(M):
        if M.size == 0:
            return 0
        s = np.linalg.svd(M, compute_uv=False)
        return int(np.sum(s > tol * max(np.linalg.norm(v), 1e-300)))
    return rank(np.hstack([Tv, Qv])) - rank(Tv)


# ------------------------------------------------------------------------------------
# Example / reference models (used in the validation suite; copy and adapt)
# ------------------------------------------------------------------------------------

def _tr(A):
    return jnp.trace(A)


def poly_det(M):
    """Determinant as an explicit polynomial (Leibniz sum) -- differentiable everywhere under jax,
    unlike jnp.linalg.det whose derivatives are NaN at singular matrices.  N! terms (N <= 7)."""
    N = M.shape[0]
    tot = 0
    for p in itertools.permutations(range(N)):
        sgn = np.linalg.det(np.eye(N)[list(p)])
        term = sgn
        for i in range(N):
            term = term * M[i, p[i]]
        tot = tot + term
    return tot


def example_SM():
    """SM with 3 generations, only the top Yukawa, V = lam (H^+ H)^2, GUT-normalised g1."""
    m = Model("SM")
    m.add_gauge("U1Y", "U1", coupling="g1", u1_norm=np.sqrt(3 / 5))
    m.add_gauge("SU2L", "SU", 2, coupling="g2")
    m.add_gauge("SU3c", "SU", 3, coupling="g3")
    m.add_weyl("Q", [("SU3c", "fund"), ("SU2L", "fund"), 3], charges={"U1Y": 1 / 6})
    m.add_weyl("uc", [("SU3c", "antifund"), 3], charges={"U1Y": -2 / 3})
    m.add_weyl("dc", [("SU3c", "antifund"), 3], charges={"U1Y": 1 / 3})
    m.add_weyl("L", [("SU2L", "fund"), 3], charges={"U1Y": -1 / 2})
    m.add_weyl("ec", [3], charges={"U1Y": 1})
    m.add_complex_scalar("H", [("SU2L", "fund")], charges={"U1Y": 1 / 2})
    m.add_yukawa("yt", lambda p, f: jnp.sum(f["uc"][:, 2] * (f["Q"][:, 0, 2] * p["H"][1] - f["Q"][:, 1, 2] * p["H"][0])))
    m.add_quartic("lam", lambda p: jnp.real(jnp.vdot(p["H"], p["H"])) ** 2)
    return m.finalize()


def example_ON(N):
    """O(N) model, V = lam (phi.phi)^2  (so lam = lambda_ON/24 for V = lambda_ON/4! (phi^2)^2)."""
    m = Model(f"O({N})")
    m.add_real_scalar("phi", [N])
    m.add_quartic("lam", lambda p: jnp.sum(p["phi"] ** 2) ** 2)
    return m.finalize()


def example_bifund_SO(N, Nf, index=0.5):
    """CCR bifundamental scalar QCD (2011.13981 eq. 2.16): SO(N)xSO(N), real N x N scalar Phi,
    Nf Majorana (Weyl) vectors of each SO(N); V = h Tr(P^T P P^T P) + f (Tr P^T P)^2."""
    m = Model(f"bifundSO({N},{Nf})")
    m.add_gauge("SOa", "SO", N, coupling="g", index=index)
    m.add_gauge("SOb", "SO", N, coupling="g", index=index)
    m.add_real_scalar("Phi", [("SOa", "fund"), ("SOb", "fund")])
    if Nf:
        m.add_weyl("psi", [("SOa", "fund"), Nf])
        m.add_weyl("chi", [("SOb", "fund"), Nf])
    m.add_quartic("h", lambda p: _tr(p["Phi"].T @ p["Phi"] @ p["Phi"].T @ p["Phi"]))
    m.add_quartic("f", lambda p: _tr(p["Phi"].T @ p["Phi"]) ** 2)
    return m.finalize()


def example_RDB(N1, N2, Nf1, Nf2, index=0.5):
    """CCR real double bifundamental model (eq. 2.31): [SO(N1)xSO(N1)] x [SO(N2)xSO(N2)];
    couplings g1,g2,h1,h2,f1,f2,zeta with V_zeta = 2 zeta Tr(P1^T P1) Tr(P2^T P2)."""
    m = Model(f"RDB({N1},{N2};{Nf1},{Nf2})")
    for i, (N, Nf) in enumerate(((N1, Nf1), (N2, Nf2)), start=1):
        m.add_gauge(f"SO{i}a", "SO", N, coupling=f"g{i}", index=index)
        m.add_gauge(f"SO{i}b", "SO", N, coupling=f"g{i}", index=index)
    for i, (N, Nf) in enumerate(((N1, Nf1), (N2, Nf2)), start=1):
        m.add_real_scalar(f"Phi{i}", [(f"SO{i}a", "fund"), (f"SO{i}b", "fund")])
        if Nf:
            m.add_weyl(f"psi{i}", [(f"SO{i}a", "fund"), Nf])
            m.add_weyl(f"chi{i}", [(f"SO{i}b", "fund"), Nf])
    for i in (1, 2):
        m.add_quartic(f"h{i}", lambda p, i=i: _tr(p[f"Phi{i}"].T @ p[f"Phi{i}"] @ p[f"Phi{i}"].T @ p[f"Phi{i}"]))
    for i in (1, 2):
        m.add_quartic(f"f{i}", lambda p, i=i: _tr(p[f"Phi{i}"].T @ p[f"Phi{i}"]) ** 2)
    m.add_quartic("zeta", lambda p: 2 * _tr(p["Phi1"].T @ p["Phi1"]) * _tr(p["Phi2"].T @ p["Phi2"]))
    return m.finalize()


def example_bifund_SU(N, Nf, yukawa=False, include_det=False):
    """SU(N)xSU(N) with complex bifundamental Phi ~ (N, Nbar) (Phi -> U Phi W^+) and Nf Dirac
    fundamentals per factor (psi ~ (N,1), chi ~ (1,N)).  V = h Tr(P^+P P^+P) + f (Tr P^+P)^2.
    yukawa=True adds  y * sum_p psic_p Phi chi_p  (needs the same Nf for both factors).
    include_det=True (N=4 only) adds lam_det * 2 Re det Phi (breaks the U(1)_B Phi -> e^{ia}Phi)."""
    m = Model(f"bifundSU({N},{Nf})")
    m.add_gauge("SUa", "SU", N, coupling="g")
    m.add_gauge("SUb", "SU", N, coupling="g")
    m.add_complex_scalar("Phi", [("SUa", "fund"), ("SUb", "antifund")])
    if Nf:
        m.add_dirac("psi", [("SUa", "fund"), Nf])
        m.add_dirac("chi", [("SUb", "fund"), Nf])
    if yukawa:
        m.add_yukawa("y", lambda p, f: jnp.einsum("ap,ab,bp->", f["psic"], p["Phi"], f["chi"]))
    PP = lambda p: p["Phi"].conj().T @ p["Phi"]
    m.add_quartic("h", lambda p: jnp.real(_tr(PP(p) @ PP(p))))
    m.add_quartic("f", lambda p: jnp.real(_tr(PP(p))) ** 2)
    if include_det:
        assert N == 4
        m.add_quartic("ldet", lambda p: 2 * jnp.real(poly_det(p["Phi"])))
    return m.finalize()


def example_CDB(N1, N2, Nf1, Nf2):
    """CCR complex double bifundamental model (eq. 2.36): [SU(N1)xSU(N1)] x [SU(N2)xSU(N2)],
    complex Phi_i ~ (N_i, Nbar_i), Nf_i Dirac fundamentals per SU factor;
    couplings g1,g2,h1,h2,f1,f2,zeta with V_zeta = 2 zeta Tr(P1^+P1) Tr(P2^+P2)."""
    m = Model(f"CDB({N1},{N2};{Nf1},{Nf2})")
    for i, N in ((1, N1), (2, N2)):
        m.add_gauge(f"SU{i}a", "SU", N, coupling=f"g{i}")
        m.add_gauge(f"SU{i}b", "SU", N, coupling=f"g{i}")
    for i, Nf in ((1, Nf1), (2, Nf2)):
        m.add_complex_scalar(f"Phi{i}", [(f"SU{i}a", "fund"), (f"SU{i}b", "antifund")])
        if Nf:
            m.add_dirac(f"psi{i}", [(f"SU{i}a", "fund"), Nf])
            m.add_dirac(f"chi{i}", [(f"SU{i}b", "fund"), Nf])
    PP = lambda p, i: p[f"Phi{i}"].conj().T @ p[f"Phi{i}"]
    for i in (1, 2):
        m.add_quartic(f"h{i}", lambda p, i=i: jnp.real(_tr(PP(p, i) @ PP(p, i))))
    for i in (1, 2):
        m.add_quartic(f"f{i}", lambda p, i=i: jnp.real(_tr(PP(p, i))) ** 2)
    m.add_quartic("zeta", lambda p: 2 * jnp.real(_tr(PP(p, 1))) * jnp.real(_tr(PP(p, 2))))
    return m.finalize()


def example_LS(Nc, Nf):
    """Litim-Sannino (1406.2337 eqs 3.1-3.6): SU(Nc), Nf Dirac fundamentals Q = (q, qc),
    complex Nf x Nf singlet H; L_Y = -y (q H^* qc + h.c.) [equivalent to y Tr(Qbar_L H Q_R + h.c.)
    up to H -> H^*], V = u Tr(H^+H H^+H) + v (Tr H^+H)^2."""
    m = Model(f"LS({Nc},{Nf})")
    m.add_gauge("SUc", "SU", Nc, coupling="g")
    m.add_dirac("q", [("SUc", "fund"), Nf])
    m.add_complex_scalar("H", [Nf, Nf])
    m.add_yukawa("y", lambda p, f: jnp.einsum("ai,ij,aj->", f["q"], p["H"].conj(), f["qc"]))
    HH = lambda p: p["H"].conj().T @ p["H"]
    m.add_quartic("u", lambda p: jnp.real(_tr(HH(p) @ HH(p))))
    m.add_quartic("v", lambda p: jnp.real(_tr(HH(p))) ** 2)
    return m.finalize()
