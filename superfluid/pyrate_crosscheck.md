# PyR@TE 3 check of the finite-N quartic beta functions

PyR@TE 3 at commit `04b219c2016f3fc4f2371d72607edc26a7e06364`, run on the model files in `pyrate_models/` ([SU(N1) x SU(N1)] x [SU(N2) x SU(N2)], one Dirac fundamental per SU factor, complex bifundamentals Phi_1, Phi_2). Its beta functions were evaluated at random couplings (gauge and all quartics, both signs), converted to 't Hooft couplings, and divided by the beta functions of the paper: the closed-form one-loop expressions and the two-loop expressions in `class1_beta2L.json`. Ratios are averaged over the sample points (6 at one loop, 8 at two loops).

| loops | (N1,N2) | ratio (h1, h2, f1, f2, zeta) | max abs(ratio - 1) |
|---|---|---|---|
| 1 | 3,2 | 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000 | 2.0e-15 |
| 1 | 3,3 | 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000 | 2.2e-15 |
| 1 | 4,3 | 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000 | 1.2e-13 |
| 1 | 5,2 | 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000 | 2.0e-15 |
| 2 | 3,2 | 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000 | 1.7e-15 |
| 2 | 3,3 | 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000 | 2.3e-15 |
| 2 | 4,3 | 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000, 1.000000000000 | 3.2e-15 |

The largest deviation of a ratio from 1 is 1.2e-13.
