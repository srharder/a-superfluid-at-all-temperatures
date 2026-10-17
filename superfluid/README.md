# Beta functions, fixed points and Table I

Python 3 with numpy, scipy, sympy, mpmath and pandas; `rgengine.py` also uses jax. Run every script from this folder.

## Table I

- `table1.py` computes Table I of the paper for all eight theories: the examples (150,783,5,26) and (300,1562,10,52)
  and the members T_5, T_9, T_13, T_21, T_41, T_101 of the family. It writes `table1.csv` and `table1_body.tex` and
  compares every entry with the printed table at the printed precision. The last column, D/D_LO, uses the exact two-loop
  couplings, the full one-loop thermal masses c_1 and c_2, and the free energy of the static modes of the large sector
  (End Matter of the paper, "Thermal corrections"); the formula is in the docstring.
- `family_fixed_points.py` writes `family_fixed_points.csv`: the ten members 5 <= n <= 101 of the family T_n
  (N_1 = n^3, N_2 = n) with the exact finite-N fixed points at one and two loops, the leading-order thermal masses and
  the residuals of the fixed-point equations. Its function `solve_theory(N1, Nf1, N2, Nf2)` works for any theory.

## Beta functions

- `class1_betas.py` contains what is specific to the theories of the paper:
  - the finite-N one-loop quartic beta functions rederived with `rgengine.py` and fitted exactly in 1/N;
  - the closed-form finite-N beta functions (one-loop quartic, two-loop gauge);
  - the planar circle, the Jacobian, the left null vector l(theta) and the angular beta function, in sympy;
  - an exact rational finite-N fixed-point solver, and the thermal analysis.
- `class1_beta2L.json` holds the finite-N two-loop quartic beta functions of Eq. (A3) as sympy strings. They were obtained
  by an exact fit to the output of `rgengine.py`. The fitting scripts are not included; `pyrate_crosscheck.py` checks the
  expressions against PyR@TE 3.
- `rgengine.py` is a general perturbative renormalization-group engine for four-dimensional gauge-Yukawa-scalar theories
  at finite N. It builds explicit group generators and evaluates the general one- and two-loop formulas (Machacek-Vaughn,
  Luo-Wang-Xiao) as tensor contractions, with no large-N input. It also fits exact polynomials in 1/N, finds fixed
  points, checks boundedness and minimizes potentials.
- `test_rgengine.py` has 31 tests: the Standard Model at two loops, QCD and Banks-Zaks, O(N) models, and the finite-N
  beta functions of the paper. Run `python -m pytest -q test_rgengine.py` (about a minute).
- `run_validation.py` and `validate_helpers.py` run a longer validation of `rgengine.py`, recorded in
  `rgengine_validation.md`.
- `pyrate_crosscheck.py` runs PyR@TE 3 on the models in `pyrate_models/` and compares its one- and two-loop beta functions
  with those of the paper. The results are in `pyrate_crosscheck.md`.
