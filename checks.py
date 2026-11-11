"""Symbolic and numerical checks of analytic results of the paper.

lambda-PF (Sec. "Self-similar lambda potential flows" and Appendix "Second-order coefficient"):
  1. light-cone exponent delta (numerical vs Eq. lightcone) and the outer acceleration tau a (Eq. accelinf);
  2. location of the critical point, s_c = 100/147 for kappa = 3, lambda = 0.7 (Eq. sc);
  3. central coefficients F_1 and F_2 from a fit of the numerical regular branch;
  4. the second-order (in F) and first-order (in u) forms of the field equation give the same flow;
  5. kappa = 3: approach to Hubble flow, max|v - r/t| -> 0.417 lambda and G - 1 = O(lambda^2);
  6. lambda = 0 (symbolic, needs sympy): reduction of Eq. FODE, the solution Eq. Flam0, and its identity with the
     spherical Rindler atmosphere outside the light cone (Eq. rindler).
Dressed Gubser flow (Sec. "Gubser flow with comoving dressing"):
  7. p ~ r^(-16/3) and r a -> 2 on tau = const, from the kinematics (u.d u) and from Euler's equation.
Usage: python checks.py   (part 6 is skipped if sympy is not installed)
"""
import numpy as np
from scipy.integrate import solve_ivp
from lambdaPF import solve, rhs, delta_analytic
from lambdaPF_source import flow
import gubser as G

print('--- 1. lambda-PF: light-cone exponent and outer acceleration')
for k, lam in [(5, 0.3), (5, 0.35), (5, 0.5), (5, 0.65), (3, -0.5), (3, 0.3), (3, 0.5)]:
    s = solve(lam, k)
    d_num = np.gradient(np.log(s['h']), s['eta'])[-1]
    d = delta_analytic(lam, k)
    vinf = d / lam
    print(f'kappa={k} lambda={lam:+.2f}: delta numerical {d_num:+.5f}, Eq. lightcone {d:+.5f};  '
          f'tau a (outer) {s["taua"][-1]:+.4f}, Eq. accelinf {-vinf / np.sqrt(1 - vinf**2):+.4f}')

print('--- 2. critical point')
s = solve(0.7, 3)
print(f'kappa=3 lambda=0.7: branch becomes sonic at s = {np.tanh(s["eta_end"])**2:.4f}; Eq. sc: s_c = 100/147 = {100 / 147:.4f}')

print('--- 3. central coefficients F_1, F_2')
for k, lam in [(5, 0.3), (3, -0.5)]:
    G0 = np.sign(lam); P = 3 + k * (lam - 1); e0 = 1e-4; Gpp0 = lam * P * G0 / 3
    sol = solve_ivp(rhs, (e0, 0.25), [G0 + 0.5 * Gpp0 * e0**2, Gpp0 * e0], args=(lam, k), method='DOP853',
                    rtol=1e-13, atol=1e-15, dense_output=True)
    eta = np.linspace(0.02, 0.12, 60)
    F = sol.sol(eta)[0] / np.cosh(eta)**lam / G0
    c = np.polyfit(np.tanh(eta)**2, F, 4)[::-1]
    F1 = k * lam * (lam - 1) / 6
    F2 = k**2 * lam * (lam - 1) / 1080 * (2 * k**2 * lam**2 - 4 * k**2 * lam + 2 * k**2 - 8 * k * lam**2
                                          + 28 * k * lam - 20 * k + 15 * lam**2 - 69 * lam + 72)
    print(f'kappa={k} lambda={lam:+.1f}: F_1 fit {c[1]:.6f}, formula {F1:.6f};  F_2 fit {c[2]:.6f}, formula {F2:.6f}')

print('--- 4. second-order (lambdaPF.solve) vs first-order (Eq. ueq, lambdaPF_source.flow) integration')
for k, lam in [(5, 0.3), (5, 0.65), (3, -0.55)]:
    s = solve(lam, k)
    eta, lnw, Y = flow(lam, k)
    sel = s['eta'] < 9
    dw = np.max(np.abs(np.log(s['w_norm'][sel]) - np.interp(s['eta'][sel], eta, lnw)))
    dv = np.max(np.abs(s['v'][sel] - np.tanh(np.interp(s['eta'][sel], eta, Y))))
    print(f'kappa={k} lambda={lam:+.2f}: max |Delta ln w| = {dw:.1e}, max |Delta v| = {dv:.1e}')

print('--- 5. kappa = 3: approach to Hubble flow')
eta = np.linspace(1e-3, 6, 600001)
dv = (1.5 * (np.sinh(eta) * np.cosh(eta) - eta) / np.sinh(eta)**2) / np.cosh(eta)**2   # |v - r/t|/lambda, first order
i = np.argmax(dv)
print(f'first order in lambda: max |v - r/t| = {dv[i]:.4f} lambda at eta = {eta[i]:.3f}')
for lam in (0.1, 0.03, 0.01, 0.003):
    so = solve(lam, 3, eta_max=6.0)
    sel = so['eta'] < 5
    print(f'lambda={lam}: max|v - r/t|/lambda = {np.max(np.abs(so["v"] - so["xi"])[sel]) / lam:.3f}, '
          f'max|G - 1|/lambda^2 = {np.max(np.abs(so["G"] - 1)[sel]) / lam**2:.2f}')

print('--- 6. lambda = 0 (symbolic)')
try:
    import sympy as sp
except ImportError:
    sp = None
    print('sympy not installed: skipped')
if sp is not None:
    s, k, C = sp.symbols('s kappa C', positive=True)
    F = sp.Function('F')
    Fp, Fpp = F(s).diff(s), F(s).diff(s, 2)
    A, B = -2 * s * Fp, 2 * (1 - s) * Fp                 # Eqs. Adef, Bdef at lambda = 0
    H2 = A**2 - 4 * s * Fp**2
    fode = k * (0 - 1) * A + 4 * s * (s - 1) * Fpp - (2 * s * (0 - 2) + 6) * Fp - (k - 1) * s * B * sp.diff(H2, s) / H2
    red = -4 * k * s * (1 - s) * Fpp + (6 * k * s - 2 * k - 4) * Fp
    print('H^2 =', sp.factor(H2), ';  Eq. FODE minus its reduced form =', sp.simplify(fode - red))
    sol = C * s**(-(k + 2) / (2 * k)) * (1 - s)**(-(k - 1) / k)     # Eq. Flam0 (for s < 1)
    res = red.subs(Fpp, sp.diff(sol, s)).subs(Fp, sol)
    print('Eq. Flam0 solves it: max |residual| =',
          max(abs(float(res.subs({k: kv, s: sv, C: 1}))) for kv in (3, 5, 7.3) for sv in (0.1, 0.5, 0.9)))
    t, r = sp.symbols('t r', positive=True)
    for kv in (3, 5):
        ss = r**2 / t**2
        Fps = -ss**sp.Rational(-(kv + 2), 2 * kv) * (ss - 1)**sp.Rational(-(kv - 1), kv)   # C = -1, r > t
        Pt, Pr = Fps * sp.diff(ss, t), Fps * sp.diff(ss, r)
        w2 = Pt**2 - Pr**2
        W = w2**sp.Rational(kv - 1, 2)
        pde = sp.diff(W * Pt, t) - sp.diff(r**2 * W * Pr, r) / r**2
        rho = sp.sqrt(r**2 - t**2)
        print(f'kappa={kv}, r > t: field-equation residual', [float(pde.subs({t: a, r: b})) for a, b in ((1, 2), (2, 5))],
              '; u^t/u^r =', sp.simplify(Pt / (-Pr)), '; w / [rho^-1 (rho/r)^(2/kappa)] =',
              sp.simplify(sp.sqrt(w2) / (rho**-1 * (rho / r)**sp.Rational(2, kv))))

print('--- 7. dressed Gubser flow, q^-1 = 4.3 fm, tau_f = 8 fm: 1/r acceleration')
q, tf = G.Q, G.TAUF
ur = lambda t, r: np.sinh(np.arctanh(G.vperp(t, r, q)))
ut = lambda t, r: np.cosh(np.arctanh(G.vperp(t, r, q)))
for r in (20, 100, 300, 1000):
    h = 1e-4
    a_kin = (ut(tf, r) * (ur(tf + h, r) - ur(tf - h, r)) + ur(tf, r) * (ur(tf, r + h) - ur(tf, r - h))) / (2 * h) / ut(tf, r)
    slope = np.log(G.pressure(tf, r * 1.01, q) / G.pressure(tf, r, q)) / np.log(1.01)
    print(f'r={r:5d} fm: r a kinematic {r * a_kin:.4f}, r a from Euler {r * G.accel(tf, r, q):.4f} (-> 2);  '
          f'd ln p / d ln r {slope:.4f} (-> -16/3)')
