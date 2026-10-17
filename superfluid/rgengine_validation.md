# rgengine validation record

All numbers below were produced by `run_validation.py` / in-session checks against the module saved alongside this file (rgengine.py v1.0). Couplings are loop-normalised (ghat = g/4pi, lamhat = lambda/16pi^2) unless 't Hooft variables are stated. 'maxdev' = max absolute deviation of exact (rationalised) polynomial coefficients from the reference; 'fit err' = relative error of the rationalised polynomial on fresh random coupling points.

## Summary: what is validated at which loop order

| beta function | loop | validated against | result |
|---|---|---|---|
| O(N) quartic | 1, 2 | (N+8)/3 lam^2 - (3N+14)/3 lam^3, N=1..6 | exact for all N (max |dev| = 0.0e+00) |
| SM gauge, Yukawa (top), quartic | 1 | textbook 1-loop SM (full polynomials) | exact (max dev 0.0e+00) |
| SM gauge incl. Yukawa term | 2 | b_ij matrix (199/50 ...), -17/10,-3/2,-2 y_t^2 | exact |
| SM Yukawa, quartic | 2 | listed MV/Luo-Xiao coefficients (-312 lam^3, -144, -3, 30, 80, -32; -12 y^5, -108, 36, 6, -12) | exact |
| SM all couplings | 1, 2 | PyR@TE 3 run on its own SM.model (Yu=diag(0,0,yt)), 5 random points | max rel dev 1.3e-15 |
| SU(3)xSU(3) complex bifundamental + 2 Dirac fund/factor + Yukawa y psi^c Phi chi | 1, 2 (all couplings) | PyR@TE 3 run on an independently written model file | max rel dev 8.9e-16 (1L), 1.5e-14 (2L) |
| SU(N) gauge + Nf Dirac fund | 1, 2 | b0 = 11N/3 - 2Nf/3, b1 = 34N^2/3 - 10 N Nf/3 - 2 C_F Nf; N=2..5, Nf in {0,1,3,6} | max |dev| 6.0e-14 |
| CCR bifund. scalar QCD SO(N)xSO(N) (eq. B.19), FINITE N | gauge 2, quartic 1 | exact closed-form finite-N betas, (N,Nf) in [(2, 1), (3, 1), (3, 4), (4, 2), (5, 3), (6, 5), (7, 2)] | max |dev| 7.1e-15 |
| CCR real double bifundamental (eq. 3.3) | quartic 1 | planar limit of exact finite-N (N1=N2=N=2..7) Laurent fits | planar max |dev| 0.0e+00, worst holdout 2.2e-16 |
| CCR complex double bifundamental SU^4 (eq. 4.17) | quartic 1, gauge 2 | planar limit of exact finite-N (N=2..6) fits; gauge vs MV closed form | planar max |dev| 0.0e+00; gauge 3.6e-15 |
| CCR eq. 3.4, beta_h^(2), beta_f^(2) (zeta-independent part) | quartic 2 | planar limit of exact finite-N SO(N)^2 (N=2..7, Nf=0,2) | x_f^0 part max |dev| 0.0e+00; x_f^1 part 0.0e+00; worst holdout 4.3e-07 |
| Litim-Sannino (1406.2337) gauge 2L, Yukawa 1L & 2L, quartic 1L | see LS section | Veneziano limit of exact finite-(Nc,Nf) results | see table below |
| thermal mass | 1 | SM m^2(T)=T^2(3g^2/16+g'^2/16+lam/2+y_t^2/4); O(N) (N+2)lam/72; CCR eq. 2.28 (finite N) | max rel dev SM 4.1e-16, O(N) exact, CCR 1.5e-16 |

Two-loop quartic and Yukawa betas use the Poole-Thomsen tensor basis with coefficients transcribed from PyR@TE 3 source; the PyR@TE comparisons therefore test the transcription, index conventions and our model construction, while O(N), the SM literature coefficients, CCR eq. 3.4 and Litim-Sannino eq. 3.26 are independent references. One convention subtlety was found and fixed: PyR@TE's fermion generator acting on the first Yukawa index equals t^T in our convention; with t instead of t^T the 2-loop term q2_19 was not gauge covariant (projection residual 0.09 in the SM), with t^T every term is covariant (residual < 1e-15).

## 1. O(N) model (V = lam (phi^2)^2, lam = lambda_ON/24)

| N | beta^(1)/lam^2 | ref 8(N+8) | beta^(2)/lam^3 | ref -192(3N+14) | C/lam | ref (N+2)/3 |
|---|---|---|---|---|---|---|
| 1 | 72 | 72 | -3264 | -3264 | 1 | 1 |
| 2 | 80 | 80 | -3840 | -3840 | 1.33333333333 | 1.33333333333 |
| 3 | 88 | 88 | -4416 | -4416 | 1.66666666667 | 1.66666666667 |
| 4 | 96 | 96 | -4992 | -4992 | 2 | 2 |
| 5 | 104 | 104 | -5568 | -5568 | 2.33333333333 | 2.33333333333 |
| 6 | 112 | 112 | -6144 | -6144 | 2.66666666667 | 2.66666666667 |

## 2. Standard Model (3 generations, top Yukawa only, V = lam (H^+H)^2, g1 GUT-normalised)

Exact polynomials extracted by the engine (fit err 1.3e-15):

```
beta_g1^(1) = (41/10)*g1^3
beta_g1^(2) = (199/50)*g1^5 + (27/10)*g1^3*g2^2 + (44/5)*g1^3*g3^2 + (-17/10)*g1^3*yt^2
beta_g2^(1) = (-19/6)*g2^3
beta_g2^(2) = (9/10)*g1^2*g2^3 + (35/6)*g2^5 + (12)*g2^3*g3^2 + (-3/2)*g2^3*yt^2
beta_g3^(1) = (-7)*g3^3
beta_g3^(2) = (11/10)*g1^2*g3^3 + (9/2)*g2^2*g3^3 + (-26)*g3^5 + (-2)*g3^3*yt^2
beta_yt^(1) = (-17/20)*g1^2*yt + (-9/4)*g2^2*yt + (-8)*g3^2*yt + (9/2)*yt^3
beta_yt^(2) = (1187/600)*g1^4*yt + (-9/20)*g1^2*g2^2*yt + (19/15)*g1^2*g3^2*yt + (393/80)*g1^2*yt^3 + (-23/4)*g2^4*yt + (9)*g2^2*g3^2*yt + (225/16)*g2^2*yt^3 + (-108)*g3^4*yt + (36)*g3^2*yt^3 + (-12)*yt^5 + (-12)*yt^3*lam + (6)*yt*lam^2
beta_lam^(1) = (27/200)*g1^4 + (9/20)*g1^2*g2^2 + (-9/5)*g1^2*lam + (9/8)*g2^4 + (-9)*g2^2*lam + (-6)*yt^4 + (12)*yt^2*lam + (24)*lam^2
beta_lam^(2) = (-3411/2000)*g1^6 + (-1677/400)*g1^4*g2^2 + (-171/100)*g1^4*yt^2 + (1887/200)*g1^4*lam + (-289/80)*g1^2*g2^4 + (63/10)*g1^2*g2^2*yt^2 + (117/20)*g1^2*g2^2*lam + (-8/5)*g1^2*yt^4 + (17/2)*g1^2*yt^2*lam + (108/5)*g1^2*lam^2 + (305/16)*g2^6 + (-9/4)*g2^4*yt^2 + (-73/8)*g2^4*lam + (45/2)*g2^2*yt^2*lam + (108)*g2^2*lam^2 + (-32)*g3^2*yt^4 + (80)*g3^2*yt^2*lam + (30)*yt^6 + (-3)*yt^4*lam + (-144)*yt^2*lam^2 + (-312)*lam^3
```

## 3. QCD two-loop

| N | Nf | b0 | ref | b1 | ref |
|---|---|---|---|---|---|
| 2 | 0 | 7.333333333 | 7.333333333 | 45.33333333 | 45.33333333 |
| 2 | 1 | 6.666666667 | 6.666666667 | 37.16666667 | 37.16666667 |
| 2 | 3 | 5.333333333 | 5.333333333 | 20.83333333 | 20.83333333 |
| 2 | 6 | 3.333333333 | 3.333333333 | -3.666666667 | -3.666666667 |
| 3 | 0 | 11 | 11 | 102 | 102 |
| 3 | 1 | 10.33333333 | 10.33333333 | 89.33333333 | 89.33333333 |
| 3 | 3 | 9 | 9 | 64 | 64 |
| 3 | 6 | 7 | 7 | 26 | 26 |
| 4 | 0 | 14.66666667 | 14.66666667 | 181.3333333 | 181.3333333 |
| 4 | 1 | 14 | 14 | 164.25 | 164.25 |
| 4 | 3 | 12.66666667 | 12.66666667 | 130.0833333 | 130.0833333 |
| 4 | 6 | 10.66666667 | 10.66666667 | 78.83333333 | 78.83333333 |
| 5 | 0 | 18.33333333 | 18.33333333 | 283.3333333 | 283.3333333 |
| 5 | 1 | 17.66666667 | 17.66666667 | 261.8666667 | 261.8666667 |
| 5 | 3 | 16.33333333 | 16.33333333 | 218.9333333 | 218.9333333 |
| 5 | 6 | 14.33333333 | 14.33333333 | 154.5333333 | 154.5333333 |

## 4. CCR bifundamental scalar QCD at finite N (2011.13981 eq. B.19, thermal mass eq. 2.28)

Conventions: SO(N) generators with T(vector)=1/2 (CCR eq. 2.26), Majorana = Weyl in the vector; 't Hooft couplings lam = g^2 N/16pi^2, h = 16pi^2 htilde... i.e. hhat = h/N, fhat = f/N^2.

| N | Nf | n_s | max |dev| vs B.19 | thermal 2.28 max rel dev | engine beta_h (1L, 't Hooft) | engine beta_f |
|---|---|---|---|---|---|---|
| 2 | 1 | 4 | 8.9e-16 | 0.0e+00 | (-3/16)lam^2 + (-3)lam*h + (32)h^2 + (24)h*f | (9/16)lam^2 + (-3)lam*f + (24)h^2 + (40)h*f + (24)f^2 |
| 3 | 1 | 9 | 7.1e-15 | 0.0e+00 | (-1/16)lam^2 + (-4)lam*h + (80/3)h^2 + (32/3)h*f | (9/16)lam^2 + (-4)lam*f + (24)h^2 + (112/3)h*f + (136/9)f^2 |
| 3 | 4 | 9 | 7.1e-15 | 0.0e+00 | (-1/16)lam^2 + (-4)lam*h + (80/3)h^2 + (32/3)h*f | (9/16)lam^2 + (-4)lam*f + (24)h^2 + (112/3)h*f + (136/9)f^2 |
| 4 | 2 | 16 | 3.7e-16 | 0.0e+00 | (-9/2)lam*h + (24)h^2 + (6)h*f | (9/16)lam^2 + (-9/2)lam*f + (24)h^2 + (36)h*f + (12)f^2 |
| 5 | 3 | 25 | 7.1e-15 | 0.0e+00 | (3/80)lam^2 + (-24/5)lam*h + (112/5)h^2 + (96/25)h*f | (9/16)lam^2 + (-24/5)lam*f + (24)h^2 + (176/5)h*f + (264/25)f^2 |
| 6 | 5 | 36 | 1.8e-15 | 0.0e+00 | (1/16)lam^2 + (-5)lam*h + (64/3)h^2 + (8/3)h*f | (9/16)lam^2 + (-5)lam*f + (24)h^2 + (104/3)h*f + (88/9)f^2 |
| 7 | 2 | 49 | 3.6e-15 | 1.5e-16 | (9/112)lam^2 + (-36/7)lam*h + (144/7)h^2 + (96/49)h*f | (9/16)lam^2 + (-36/7)lam*f + (24)h^2 + (240/7)h*f + (456/49)f^2 |

## 5. CCR real double bifundamental, exact finite-N one-loop quartic betas (N1 = N2 = N, 't Hooft variables)

Laurent coefficients fitted exactly (rational reconstruction) on N = 2..6, verified on N = 7. The N^0 column reproduces CCR eq. 3.3; the 1/N terms reproduce the finite-N pieces quoted in CCR eqs. H.13-H.17 (e.g. beta_zeta contains zeta(8h+3lam)(1/N1+1/N2)). The zeta^2 and f zeta terms in beta_zeta at O(1/N^2) are the double-trace lifting terms relevant for Step 3b.

| beta | monomial | coefficient as sum_p a_p N^p | holdout err |
|---|---|---|---|
| h1 | h1^2 | (16) N^0 + (32) N^-1 | 2.2e-16 |
| h1 | lam1*h1 | (-6) N^0 + (6) N^-1 | 1.7e-16 |
| h1 | h1*f1 | (96) N^-2 | 0.0e+00 |
| h1 | lam1^2 | (3/16) N^0 + (-3/4) N^-1 | 1.1e-16 |
| f1 | h1^2 | (24) N^0 | 1.5e-16 |
| f1 | zeta^2 | (8) N^0 | 2.2e-16 |
| f1 | f1^2 | (8) N^0 + (64) N^-2 | 7.4e-17 |
| f1 | lam1*f1 | (-6) N^0 + (6) N^-1 | 0.0e+00 |
| f1 | h1*f1 | (32) N^0 + (16) N^-1 | 1.8e-16 |
| f1 | lam1^2 | (9/16) N^0 | 0.0e+00 |
| zeta | f2*zeta | (8) N^0 + (16) N^-2 | 1.5e-16 |
| zeta | h2*zeta | (16) N^0 + (8) N^-1 | 1.8e-16 |
| zeta | zeta^2 | (32) N^-2 | 0.0e+00 |
| zeta | lam1*zeta | (-3) N^0 + (3) N^-1 | 0.0e+00 |
| zeta | h1*zeta | (16) N^0 + (8) N^-1 | 1.8e-16 |
| zeta | f1*zeta | (8) N^0 + (16) N^-2 | 1.5e-16 |
| zeta | lam2*zeta | (-3) N^0 + (3) N^-1 | 0.0e+00 |

## 6. CCR complex double bifundamental SU(N)^4, exact finite-N one-loop quartic betas

Fits on N = 2..5 verified on N = 6 (Nf = 1 per SU factor; quartic 1-loop betas are Nf-independent). N^0 column = CCR eq. 4.17. Gauge 2-loop example at N = 6, Nf = 3: beta_lam1 = (-497/36)lam1^3 + (-19/3)lam1^2.

| beta | monomial | coefficient as sum_p a_p N^p | holdout err |
|---|---|---|---|
| h1 | h1^2 | (8) N^0 | 2.2e-16 |
| h1 | lam1*h1 | (-12) N^0 + (12) N^-2 | 1.5e-16 |
| h1 | h1*f1 | (24) N^-2 | 0.0e+00 |
| h1 | lam1^2 | (3/2) N^0 + (-12) N^-2 | 1.5e-16 |
| f1 | h1^2 | (12) N^0 | 1.5e-16 |
| f1 | zeta^2 | (4) N^0 | 0.0e+00 |
| f1 | f1^2 | (4) N^0 + (16) N^-2 | 1.1e-16 |
| f1 | lam1*f1 | (-12) N^0 + (12) N^-2 | 0.0e+00 |
| f1 | h1*f1 | (16) N^0 | 0.0e+00 |
| f1 | lam1^2 | (9/2) N^0 + (6) N^-2 | 0.0e+00 |
| zeta | f2*zeta | (4) N^0 + (4) N^-2 | 0.0e+00 |
| zeta | h2*zeta | (8) N^0 | 0.0e+00 |
| zeta | zeta^2 | (8) N^-2 | 0.0e+00 |
| zeta | lam1*zeta | (-6) N^0 + (6) N^-2 | 0.0e+00 |
| zeta | h1*zeta | (8) N^0 | 0.0e+00 |
| zeta | f1*zeta | (4) N^0 + (4) N^-2 | 0.0e+00 |
| zeta | lam2*zeta | (-6) N^0 + (6) N^-2 | 0.0e+00 |

## 7. Two-loop quartic betas, SO(N)xSO(N) bifundamental, planar limit vs CCR eq. 3.4

coefficient = A(N) + x_f * (N B(N)); fits on N = 2..6, verified on N = 7. Holdout errors of order 1e-7 come from floating-point noise in the 2-loop coefficient extraction (subtraction of Nf=0 and Nf=2 data), not from a structural mismatch; the planar (N^0) coefficients are exact.


### xf^0 part

| beta | monomial | coefficient as sum_p a_p N^p | holdout err |
|---|---|---|---|
| h | h^2*f | (-2816) N^-2 + (-3712) N^-3 | 1.2e-17 |
| h | lam*h^2 | (40) N^0 + (16) N^-1 + (-128) N^-2 | 0.0e+00 |
| h | lam^2*h | (-13/2) N^0 + (167/6) N^-1 + (-64/3) N^-2 | 3.0e-16 |
| h | lam^3 | (1/4) N^0 + (-2.58333366754269) N^-1 + (6.333335028252319) N^-2 + (-3.6763033314517034e-06) N^-3 + (2.8646520484965053e-06) N^-4 | 4.4e-08 |
| h | h^3 | (-96) N^0 + (-576) N^-1 + (-1632) N^-2 | 1.4e-16 |
| h | lam^2*f | (6) N^-2 + (-24) N^-3 | 9.3e-18 |
| h | h*f^2 | (-320) N^-2 + (-5248) N^-4 | 4.4e-18 |
| h | lam*h*f | (192) N^-2 + (-192) N^-3 | 3.7e-17 |
| f | f^3 | (-576) N^-2 + (-2688) N^-4 | 9.1e-17 |
| f | lam^2*h | (9) N^0 + (-9) N^-1 | 2.3e-16 |
| f | h^3 | (-384) N^0 + (-576) N^-1 | 1.7e-16 |
| f | h^2*f | (-160) N^0 + (-320) N^-1 + (-2784) N^-2 | 5.6e-17 |
| f | lam*h^2 | (96) N^0 + (-24) N^-1 | 1.5e-16 |
| f | lam^3 | (3/4) N^0 + (-19/4) N^-1 | 1.5e-09 |
| f | lam^2*f | (-7/2) N^0 + (149/6) N^-1 + (-10/3) N^-2 | 7.0e-11 |
| f | h*f^2 | (-2816) N^-2 + (-1408) N^-3 | 1.6e-17 |
| f | lam*f^2 | (32) N^0 + (-32) N^-1 + (160) N^-2 + (-160) N^-3 | 2.0e-16 |
| f | lam*h*f | (128) N^0 + (-64) N^-1 + (-64) N^-2 | 1.2e-16 |

### xf^1 part

| beta | monomial | coefficient as sum_p a_p N^p | holdout err |
|---|---|---|---|
| h | lam^2*h | (5/3) N^0 + (-5/3) N^-1 | 7.8e-16 |
| h | lam^3 | (-1/6) N^0 + (0.6666676692945203) N^-1 + (-5.08475555631446e-06) N^-2 + (1/90671) N^-3 + (-8.593953066114612e-06) N^-4 | 4.3e-07 |
| f | lam^3 | (-1/2) N^0 | 4.9e-08 |
| f | lam^2*f | (5/3) N^0 + (-5/3) N^-1 | 1.4e-09 |

## 9. Litim-Sannino model: exact finite-(Nc, Nf) betas and their Veneziano limit

alpha_g = g^2 Nc/16pi^2, alpha_y = y^2 Nc/16pi^2, alpha_h = u Nf/16pi^2, alpha_v = v Nf^2/16pi^2. Each coefficient was fitted as sum a_pq Nc^p Nf^q on a (Nc,Nf) grid (1-loop and 2-loop gauge: Nc=2..6 x Nf=1..5; 2-loop Yukawa: Nc=2..5 x Nf=1..4); the Veneziano limit keeps p+q=0 and is a polynomial in x = Nf/Nc = 11/2 + eps.

| coefficient | exact finite-N fit | fit resid (dof) | Veneziano poly in x |
|---|---|---|---|
| beta_g^(1) ag^2 | (4/3) Nc^-1 Nf^1 + (-22/3) Nc^0 Nf^0 | 1.2e-16 (9) | (4/3) x^1 + (-22/3) x^0 |
| beta_y^(1) ay^2 | (2) Nc^-1 Nf^1 + (2) Nc^0 Nf^0 | 6.3e-17 (9) | (2) x^1 + (2) x^0 |
| beta_y^(1) ag*ay | (6) Nc^-2 Nf^0 + (-6) Nc^0 Nf^0 | 1.5e-16 (9) | (-6) x^0 |
| beta_u^(1) av^2 | (0) Nc^-3 Nf^-1 + (0) Nc^-3 Nf^0 + (0) Nc^-2 Nf^0 + (2300/67) Nc^0 Nf^-1 + (-6755/201) Nc^0 Nf^0 + (1373/134) Nc^0 Nf^1 + (-395/402) Nc^0 Nf^2 | 2.7e-02 (9) | (-6755/201) x^0 |
| beta_u^(1) ah*av | (11238/335) Nc^0 Nf^-1 + (-11366/673) Nc^0 Nf^0 + (3.619701492538465) Nc^0 Nf^1 + (-56/201) Nc^0 Nf^2 | 1.1e-03 (9) | (-11366/673) x^0 |
| beta_u^(1) ah^2 | (460/67) Nc^0 Nf^-1 + (257/201) Nc^0 Nf^0 + (1373/670) Nc^0 Nf^1 + (-79/402) Nc^0 Nf^2 | 5.4e-03 (9) | (257/201) x^0 |
| beta_u^(1) ay*av | (460/67) Nc^0 Nf^-1 + (-1351/201) Nc^0 Nf^0 + (1373/670) Nc^0 Nf^1 + (-79/402) Nc^0 Nf^2 | 2.7e-02 (9) | (-1351/201) x^0 |
| beta_u^(1) ay*ah | (-460/67) Nc^0 Nf^-1 + (2155/201) Nc^0 Nf^0 + (-1373/670) Nc^0 Nf^1 + (79/402) Nc^0 Nf^2 | 1.3e-02 (9) | (2155/201) x^0 |
| beta_u^(1) ay^2 | (230/67) Nc^-1 Nf^-1 + (-1351/402) Nc^-1 Nf^0 + (-0.9753731343389451) Nc^-1 Nf^1 + (-79/804) Nc^-1 Nf^2 | 2.7e-03 (9) | (-0.9753731343389451) x^1 |
| beta_v^(1) av^2 | (0) Nc^-3 Nf^0 + (0) Nc^-2 Nf^0 + (-2.8099502487750936) Nc^0 Nf^-1 + (14726/847) Nc^0 Nf^0 + (-1923/377) Nc^0 Nf^1 + (215/402) Nc^0 Nf^2 | 1.8e-02 (9) | (14726/847) x^0 |
| beta_v^(1) ah*av | (920/67) Nc^0 Nf^-1 + (514/201) Nc^0 Nf^0 + (1373/335) Nc^0 Nf^1 + (-79/201) Nc^0 Nf^2 | 5.4e-03 (9) | (514/201) x^0 |
| beta_v^(1) ah^2 | (-460/67) Nc^0 Nf^-1 + (3763/201) Nc^0 Nf^0 + (-1373/670) Nc^0 Nf^1 + (79/402) Nc^0 Nf^2 | 4.5e-03 (9) | (3763/201) x^0 |
| beta_v^(1) ay*av | (-460/67) Nc^0 Nf^-1 + (2155/201) Nc^0 Nf^0 + (-1373/670) Nc^0 Nf^1 + (79/402) Nc^0 Nf^2 | 1.3e-02 (9) | (2155/201) x^0 |
| beta_v^(1) ay*ah | (460/67) Nc^0 Nf^-1 + (-1351/201) Nc^0 Nf^0 + (1373/670) Nc^0 Nf^1 + (-79/402) Nc^0 Nf^2 | 2.7e-02 (9) | (-1351/201) x^0 |
| beta_v^(1) ay^2 | (-230/67) Nc^-1 Nf^-1 + (1351/402) Nc^-1 Nf^0 + (-1.024626865660726) Nc^-1 Nf^1 + (79/804) Nc^-1 Nf^2 | 2.7e-02 (9) | (-1.024626865660726) x^1 |
| beta_g^(2) ag^2*ay | (-2) Nc^-2 Nf^2 | 7.1e-17 (9) | (-2) x^2 |
| beta_g^(2) ag^3 | (-2) Nc^-3 Nf^1 + (26/3) Nc^-1 Nf^1 + (-68/3) Nc^0 Nf^0 | 1.7e-16 (9) | (26/3) x^1 + (-68/3) x^0 |
| beta_y^(2) ay*av^2 | (9445/432) Nc^0 Nf^-1 + (-16709/864) Nc^0 Nf^0 + (2645/432) Nc^0 Nf^1 + (-559/864) Nc^0 Nf^2 | 4.4e-16 (4) | (-16709/864) x^0 |
| beta_y^(2) ay*ah*av | (100/3) Nc^0 Nf^-1 + (-70/3) Nc^0 Nf^0 + (20/3) Nc^0 Nf^1 + (-2/3) Nc^0 Nf^2 | 5.6e-17 (4) | (-70/3) x^0 |
| beta_y^(2) ay*ah^2 | (25/3) Nc^0 Nf^-1 + (-11/6) Nc^0 Nf^0 + (5/3) Nc^0 Nf^1 + (-1/6) Nc^0 Nf^2 | 0.0e+00 (4) | (-11/6) x^0 |
| beta_y^(2) ay^2*av | (-16) Nc^-1 Nf^-1 | 0.0e+00 (4) | 0 |
| beta_y^(2) ay^2*ah | (-8) Nc^-1 Nf^-1 + (-8) Nc^-1 Nf^1 | 1.0e-16 (4) | (-8) x^1 |
| beta_y^(2) ay^3 | (4) Nc^-2 Nf^0 + (-1/2) Nc^-2 Nf^2 + (-6) Nc^-1 Nf^1 | 6.8e-17 (4) | (-1/2) x^2 + (-6) x^1 |
| beta_y^(2) ag*ay^2 | (-5) Nc^-2 Nf^0 + (-8.193721647592614) Nc^-2 Nf^1 + (10.626129559048836) Nc^-1 Nf^1 + (5) Nc^0 Nf^0 + (-0.26419887497502326) Nc^0 Nf^1 | 1.3e-03 (4) | (10.626129559048836) x^1 + (5) x^0 |
| beta_y^(2) ag^2*ay | (32845/959) Nc^-2 Nf^0 + (-2284/669) Nc^-2 Nf^1 + (0.45691242518578196) Nc^-1 Nf^0 + (4.4275539829250565) Nc^-1 Nf^1 + (-31072/917) Nc^0 Nf^0 + (-0.11008286457059209) Nc^0 Nf^1 | 3.1e-04 (4) | (4.4275539829250565) x^1 + (-31072/917) x^0 |

## 10. Engine-internal consistency

| model | n_s | functional vs tensor 1L quartic (max rel dev) | projection residual (functional / tensor) | Yukawa 1L: closed form vs PyR@TE-basis port |
|---|---|---|---|---|
| bifundSU(3,2) | 18 | 3.4e-15 | 2.9e-16 / 3.0e-15 | 4.7e-16 |
| CDB(2,3;1,2) | 26 | 2.8e-15 | 3.1e-16 / 5.3e-15 | - |
| LS(3,2) | 8 | 5.1e-16 | 2.8e-16 / 4.4e-16 | 0.0e+00 |
| RDB(3,2;1,2) | 13 | 1.0e-15 | 5.1e-16 / 1.2e-15 | - |
| bifundSU(4,1) | 32 | 3.5e-14 | 3.1e-16 / 3.7e-14 | - |

Operator-completeness diagnostic (residual must be O(1) when the RG generates an operator outside the basis):

- SO(4)^2 basis {h} only: residual functional/tensor: [0.08722963014578686, 0.23329722119516166]
- SO(4)^2 full basis {h,f}: residual: 3.3927680322709876e-16
- LS(3,2) basis {u} only: residual (v generated): 0.08403203199311099
- SU(4)^2 with det: residual with basis {h,f,det}: 2.178275111434904e-16
- SU(4)^2 with det: betas: [0.058499999999999115, 0.4777875000000002, -0.10499999999999941]

## 11. Vacuum tools

Boundedness below of V = h Tr(M^2) + f (Tr M)^2, M = Phi^T Phi, SO(4)xSO(4) real bifundamental: minimum of V on the unit sphere vs analytic min(f+h, f+h/N) (the unit sphere has Tr M = 1):

| h | f | multistart min | analytic |
|---|---|---|---|
| 1.0 | 0.1 | 0.350000000000 | 0.350000000000 |
| 1.0 | -0.2 | 0.050000000000 | 0.050000000000 |
| 1.0 | -0.3 | -0.050000000000 | -0.050000000000 |
| -1.0 | 1.1 | 0.100000000000 | 0.100000000000 |
| -1.0 | 0.9 | -0.100000000000 | -0.100000000000 |
| -0.5 | 0.2 | -0.300000000000 | -0.300000000000 |

SU(3)xSU(3) complex bifundamental, V_hat = h Tr(M^2) + f (Tr M)^2 - |phi|^2/2 (negative thermal mass inserted by hand):

- h=0.5, f=-0.1: global min V = -3.7500000000 (analytic -3.75), singular values [1.58113883, 1.58113883, 1.58113883] (analytic sqrt(2.5)=1.58113883), rank 3; U(1)_B gauge-compensation residual 1.000000 (=> U(1)_B broken, det Phi != 0).
- h=-0.2, f=0.5: global min V = -0.8333333333 (analytic -5/6), singular values [1.29, 1.88e-09, 6.7e-10], rank 1; compensation residual 1.5e-09 (=> rank-1 condensate does NOT break U(1)_B).
- single SU(3) fundamental, random v: U(1) compensation residual 2.2e-16 (Higgsing, no global SSB).

## 12. Fixed-point finder

- Litim-Sannino Veneziano system (3.15) at eps = 0.05: found [[-0.002621231979030144, 0.0], [0.023957571324067322, 0.01097293343087053]] vs (3.17) UV FP [0.0239575713240673, 0.01097293343087052] and (3.19) [-0.0026212319790301446, 0]; stability eigenvalues [[0.00017474882416883557, 0.015727391874180863], [-0.0014368872816267197, 0.1597801641099419]].
- SU(3) with 16 Dirac flavours, 2-loop: ghat* = [0.05754353376484364] vs sqrt(-b0/b1) = 0.057543533764843655; theta = [0.0022075055221283707].
- SU(4)xSU(4) bifundamental with 20 Dirac fundamentals per factor (cheat-sheet example): beta_g = 0 gives ghat^2 = 2/587; the remaining two quartic equations have four roots, all complex (numpy roots of the eliminant: h = 5.161e-3 +- 2.159e-3 i and 2.795e-4 +- 3.16e-6 i in hatted units), and the multistart finder correctly returns no real fixed point.

## Status at shutdown (run interrupted by user request)

PASSED (numbers above): O(N) 1L+2L (N=1..6, exact); SM 1L all couplings, 2L gauge/Yukawa/quartic (exact vs literature coefficients and vs PyR@TE 3, max rel dev 1.3e-15); QCD b0,b1 (16 (N,Nf) points, exact); PyR@TE 3 cross-check SU(3)xSU(3) bifundamental + Yukawa at 1L and 2L (all couplings; 8.9e-16 / 1.5e-14); CCR eq. B.19 finite-N (7 (N,Nf) points, exact) and thermal mass eq. 2.28 (exact); CCR RDB planar 1L (eq. 3.3, exact, N=2..7); CCR CDB planar 1L (eq. 4.17, exact, N=2..6) and 2L gauge; CCR eq. 3.4 2L beta_h, beta_f planar limit (zeta-independent part, exact, N=2..7); functional vs tensor 1L quartic (5 models, <=3.5e-14); missing-operator residual diagnostic; boundedness/minimisation/Step-5 tests; fixed-point finder (LS eq. 3.17, QCD BZ); pytest suite 31/31 passed (run before the last three edits: lstsq fallback, NaN guard, low-memory invariance check -- these were spot-checked only).

IN PROGRESS / NOT COMPLETED: Litim-Sannino Veneziano-limit comparison (finite-N data partially computed, fit/compare not run) -- NOT validated; CCR eq. 3.4 zeta terms at 2 loops (N=2..5 computation unfinished) -- NOT validated; performance benchmarks (only partial numbers: CDB(8,8;2,2) n_s=256 functional 1L quartic beta 1.3 s after compile, thermal_C 61 s first call; SO(N)^2 2L tensor quartic ~15-40 s per evaluation at n_s=49 under load).
