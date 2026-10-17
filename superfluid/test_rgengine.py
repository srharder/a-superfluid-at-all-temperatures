"""pytest suite for rgengine (fast subset of the validation in rgengine_validation.md).
Run:  python -m pytest -q test_rgengine.py      (~3-6 min on a laptop CPU)"""
import numpy as np
import jax.numpy as jnp
import pytest
import rgengine as rg

LOOP = 16 * np.pi ** 2


# ---------------------------------------------------------------- group theory
@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_su_casimirs(N):
    F = rg.GaugeFactor("G", "SU", N)
    assert len(F.fund) == N * N - 1
    assert np.isclose(F.C2G, N)
    C = rg.casimir(F.fund)
    assert np.allclose(C, (N * N - 1) / (2 * N) * np.eye(N))


@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_so_casimirs(N):
    F = rg.GaugeFactor("G", "SO", N)  # index 1/2 (CCR convention)
    assert len(F.fund) == N * (N - 1) // 2
    assert np.isclose(F.C2G, (N - 2) / 2)
    assert np.allclose(rg.casimir(F.fund), (N - 1) / 4 * np.eye(N))


@pytest.mark.parametrize("N", [2, 4, 6])
def test_sp_casimirs(N):
    F = rg.GaugeFactor("G", "SP", N)
    n = N // 2
    assert len(F.fund) == n * (2 * n + 1)
    assert np.isclose(F.C2G, n + 1)


def test_two_index_reps():
    F = rg.GaugeFactor("G", "SU", 4)
    gs, es = F.rep("sym2")
    ga, ea = F.rep("antisym2")
    # Casimirs C2(S) = (N-1)(N+2)/N, C2(A) = (N-2)(N+1)/N with T(fund)=1/2
    assert np.allclose(rg.casimir(gs), 3 * 6 / 4 * np.eye(10))
    assert np.allclose(rg.casimir(ga), 2 * 5 / 4 * np.eye(6))


# ---------------------------------------------------------------- O(N)
@pytest.mark.parametrize("N", [1, 2, 4])
def test_ON_two_loop(N):
    m = rg.example_ON(N)
    lam = 0.013
    b1 = m.beta_quartic(np.array([lam]), 1, "tensor")[0][0]
    b12 = m.beta_quartic(np.array([lam]), 2, "tensor")[0][0]
    assert np.isclose(b1, 8 * (N + 8) * lam ** 2, rtol=1e-12)
    assert np.isclose(b12 - b1, -192 * (3 * N + 14) * lam ** 3, rtol=1e-10)
    bf = m.beta_quartic(np.array([lam]), 1, "functional")[0][0]
    assert np.isclose(bf, b1, rtol=1e-10)
    C = m.thermal_C(np.array([lam]), raw=False)
    assert np.allclose(C, lam * (N + 2) / 3 * np.eye(N))


# ---------------------------------------------------------------- QCD
@pytest.mark.parametrize("N,Nf", [(3, 0), (3, 6), (2, 5), (4, 3)])
def test_qcd_two_loop(N, Nf):
    m = rg.Model()
    m.add_gauge("SU", "SU", N, coupling="g")
    if Nf:
        m.add_dirac("q", [("SU", "fund"), Nf])
    m.finalize()
    g = 0.21
    b1 = m.beta_gauge([g], 1)[0]
    b2 = m.beta_gauge([g], 2)[0] - b1
    CF = (N * N - 1) / (2 * N)
    assert np.isclose(b1, -(11 * N / 3 - 2 * Nf / 3) * g ** 3)
    assert np.isclose(b2, -(34 * N * N / 3 - 10 * N * Nf / 3 - 2 * CF * Nf) * g ** 5)


# ---------------------------------------------------------------- SM
@pytest.fixture(scope="module")
def sm():
    return rg.example_SM()


def test_sm_one_loop(sm):
    g1, g2, g3, yt, lam = c = np.array([0.35, 0.52, 0.9, 0.8, 0.11])
    b = sm.beta(c, loops={"gauge": 1, "yukawa": 1, "quartic": 1})
    ref = [41 / 10 * g1 ** 3, -19 / 6 * g2 ** 3, -7 * g3 ** 3,
           yt * (9 / 2 * yt ** 2 - 8 * g3 ** 2 - 9 / 4 * g2 ** 2 - 17 / 20 * g1 ** 2),
           24 * lam ** 2 + 12 * lam * yt ** 2 - 6 * yt ** 4 - lam * (9 / 5 * g1 ** 2 + 9 * g2 ** 2)
           + 27 / 200 * g1 ** 4 + 9 / 20 * g1 ** 2 * g2 ** 2 + 9 / 8 * g2 ** 4]
    assert np.allclose(b, ref, rtol=1e-12)


def test_sm_two_loop_gauge(sm):
    g1, g2, g3, yt, lam = c = np.array([0.35, 0.52, 0.9, 0.8, 0.11])
    d = sm.beta_gauge(c, 2) - sm.beta_gauge(c, 1)
    bm = np.array([[199 / 50, 27 / 10, 44 / 5], [9 / 10, 35 / 6, 12], [11 / 10, 9 / 2, -26]])
    gg = np.array([g1, g2, g3])
    ref = gg ** 3 * (bm @ gg ** 2) - gg ** 3 * np.array([17 / 10, 3 / 2, 2]) * yt ** 2
    assert np.allclose(d, ref, rtol=1e-12)


def test_sm_two_loop_quartic_top_qcd(sm):
    # g1 = g2 = 0: beta_lam^(2) = -312 l^3 - 144 l^2 y^2 - 3 l y^4 + 30 y^6 + 80 g3^2 l y^2 - 32 g3^2 y^4
    g3, yt, lam = 0.9, 0.8, 0.11
    c = np.array([0, 0, g3, yt, lam])
    b = sm.beta(c, loops={"gauge": 2, "yukawa": 2, "quartic": 2}) - sm.beta(c, loops={"gauge": 2, "yukawa": 1, "quartic": 1})
    ref_l = -312 * lam ** 3 - 144 * lam ** 2 * yt ** 2 - 3 * lam * yt ** 4 + 30 * yt ** 6 + 80 * g3 ** 2 * lam * yt ** 2 - 32 * g3 ** 2 * yt ** 4
    ref_y = yt * (-12 * yt ** 4 - 108 * g3 ** 4 + 36 * g3 ** 2 * yt ** 2 + 6 * lam ** 2 - 12 * lam * yt ** 2)
    assert np.isclose(b[4], ref_l, rtol=1e-10)
    assert np.isclose(b[3], ref_y, rtol=1e-10)


def test_sm_thermal(sm):
    g1, g2, g3, yt, lam = c = np.array([0.35, 0.52, 0.9, 0.8, 0.11])
    C = sm.thermal_C(c)
    ref = LOOP * (3 * g2 ** 2 / 16 + 3 / 5 * g1 ** 2 / 16 + lam / 2 + yt ** 2 / 4)
    assert np.allclose(C, ref * np.eye(4))


# ---------------------------------------------------------------- CCR finite N (B.19, 2.28)
def test_ccr_B19_finite_N():
    N, Nf = 4, 3
    m = rg.example_bifund_SO(N, Nf)
    lam, h, f = 0.12, 0.05, -0.03   # 't Hooft couplings
    c = np.array([np.sqrt(lam / N), h / N, f / N ** 2])
    b = m.beta(c, loops={"gauge": 2, "quartic": 1})
    x = Nf / N
    bl = -((21 - 4 * x) / 6 - 22 / (3 * N)) * lam ** 2 + ((13 * x - 27) / 6 - (23 * x - 128) / (6 * N) - 68 / (3 * N ** 2)) * lam ** 3
    bh = 96 * f * h / N ** 2 + h ** 2 * (32 / N + 16) + h * lam * (6 / N - 6) + lam ** 2 * (3 / 16 - 3 / (4 * N))
    bf = f ** 2 * (64 / N ** 2 + 8) + f * h * (16 / N + 32) + 24 * h ** 2 + f * lam * (6 / N - 6) + 9 * lam ** 2 / 16
    got = [2 * N * c[0] * b[0], N * b[1], N ** 2 * b[2]]
    assert np.allclose(got, [bl, bh, bf], rtol=1e-11)
    C = m.thermal_C(c)
    ref = LOOP * (2 / 3 * (1 + 1 / (2 * N)) * h + 1 / 3 * (1 + 2 / N ** 2) * f + 1 / 8 * (1 - 1 / N) * lam)
    assert np.allclose(C, ref * np.eye(N * N))


# ---------------------------------------------------------------- engine internals
def test_functional_vs_tensor_with_yukawa():
    m = rg.example_bifund_SU(3, 2, yukawa=True)
    c = np.array([0.3, 0.4, 0.07, -0.02])
    bf, rf = m.beta_quartic(c, 1, "functional")
    bt, rt = m.beta_quartic(c, 1, "tensor")
    assert rf < 1e-10 and rt < 1e-10
    assert np.allclose(bf, bt, rtol=1e-9)
    gfac, y, lam = m.unpack(c)
    B1 = m.beta_yukawa_tensor1(gfac, m.Ytot(y))
    _, By = m.tensor_betas(c, 0, 1)
    assert np.allclose(B1, By, atol=1e-12)


def test_missing_operator_detected():
    m = rg.Model()
    m.add_gauge("SOa", "SO", 3, coupling="g")
    m.add_gauge("SOb", "SO", 3, coupling="g")
    m.add_real_scalar("Phi", [("SOa", "fund"), ("SOb", "fund")])
    m.add_quartic("h", lambda p: jnp.trace(p["Phi"].T @ p["Phi"] @ p["Phi"].T @ p["Phi"]))
    m.finalize()
    _, r = m.beta_quartic(np.array([0.3, 0.1]), 1, "functional")
    assert r > 1e-2          # (Tr Phi^T Phi)^2 is generated but not in the basis
    full = rg.example_bifund_SO(3, 0)
    _, r2 = full.beta_quartic(np.array([0.3, 0.1, 0.0]), 1, "functional")
    assert r2 < 1e-10


def test_gauge_invariance_check_rejects_bad_operator():
    m = rg.Model()
    m.add_gauge("SU", "SU", 3, coupling="g")
    m.add_complex_scalar("q", [("SU", "fund")])
    m.add_quartic("bad", lambda p: jnp.real(p["q"][0]) ** 4)
    with pytest.raises(ValueError):
        m.finalize()


def test_polynomial_extraction_and_laurent():
    m = rg.example_bifund_SO(3, 1)
    P, err = rg.model_beta_polynomials(m, loops={"gauge": 2, "quartic": 1})
    assert err < 1e-10
    # beta_hhat 1-loop coefficient of hhat^2 = (32/N + 16) N = 32 + 16 N (exact rational), N = 3
    assert P[1][(0, 2, 0)] == 80
    fr, e = rg.fit_laurent([2, 3, 4, 5, 6], [7 + 3 / N - 2 / N ** 2 for N in [2, 3, 4, 5, 6]], [0, -1, -2])
    assert fr == {0: 7, -1: 3, -2: -2} and e < 1e-12


# ---------------------------------------------------------------- vacuum tools
def test_step5_gauge_compensation():
    m = rg.Model()
    m.add_gauge("SU", "SU", 3, coupling="g")
    m.add_complex_scalar("q", [("SU", "fund")])
    m.add_quartic("l", lambda p: jnp.real(jnp.vdot(p["q"], p["q"])) ** 2)
    m.finalize()
    v = np.random.default_rng(0).normal(size=m.ns)
    assert rg.gauge_compensation(v, m.u1_generator({"q": 1}), m.gauge_generators_real())["residual"] < 1e-10
    mb = rg.example_bifund_SU(3, 0)
    v = mb.phi_from_views({"Phi": np.eye(3) * 0.7})
    r = rg.gauge_compensation(v, mb.u1_generator({"Phi": 1}), mb.gauge_generators_real())["residual"]
    assert r > 0.99  # full-rank square bifundamental breaks U(1)_B
    v1 = mb.phi_from_views({"Phi": np.diag([1.0, 0, 0])})
    r1 = rg.gauge_compensation(v1, mb.u1_generator({"Phi": 1}), mb.gauge_generators_real())["residual"]
    assert r1 < 1e-10  # rank-deficient: det = 0, U(1)_B compensable


def test_bounded_below_and_minimum():
    N = 3
    m = rg.example_bifund_SU(N, 0)
    for h, f, bounded in [(0.5, -0.1, True), (0.5, -0.2, False), (-0.2, 0.5, True), (-0.2, 0.1, False)]:
        bb = rg.bounded_below(m.V4(np.array([h, f])), m.ns, nstarts=24, seed=0)
        assert np.isclose(bb["min_value"], min(h + f, h / N + f) / 4, rtol=1e-4, atol=1e-7)
        assert (bb["min_value"] > 0) == bounded
    V = m.V4(np.array([0.5, -0.1]))
    mins = rg.minimize_potential(lambda x: V(x) - 0.5 * jnp.sum(x * x), m.ns, nstarts=12, seed=1)
    inv = rg.matrix_invariants(m, mins[0]["x"])
    s = inv["Phi"]["singular_values"]
    assert inv["Phi"]["rank"] == N and np.allclose(s, s[0], rtol=1e-5)


def test_fixed_point_finder_qcd_bz():
    m = rg.Model()
    m.add_gauge("SU", "SU", 3, coupling="g")
    m.add_dirac("q", [("SU", "fund"), 16])
    m.finalize()
    sols = rg.find_fixed_points(lambda x: m.beta_gauge(x, 2), 1, nstarts=20, bounds=([0.01], [0.5]),
                                keep=lambda x: x[0] > 1e-4)
    b0, b1 = 11 - 32 / 3, 102 - 38 * 16 / 3
    assert len(sols) == 1 and np.isclose(sols[0]["x"][0], np.sqrt(-b0 / b1))
    assert sols[0]["stability"][0].real > 0  # IR attractive
