"""lambda-PF with the fast-tail dressing: particle number, and dependence of the Levy fits on the cut r <= r_cut.

1. Particle number. On tau_f, n ~ r^(kappa delta - 2 beta) and u.dSigma ~ r dr, so N is finite for
   2 beta > 2 + kappa delta, while a power-law tail needs 2 beta >= 1 - delta. Together they give
   alpha_inf = 8 beta - (kappa-3) delta - 3 > 2. Printed: which cases of the paper's tables have a finite N; the
   smallest alpha_inf compatible with a finite N in the subsonic window, for several kappa; and, for Hubble flow, the
   numerical growth exponent of N(r < r_max) compared with 2 - 2 beta.
2. Cut scan (table "solIcut"): side-direction Levy fits for kappa = 5, beta = 0.55, lambda = 0.3, 0.4, 0.5, 0.65,
   m_T = 0.2 and 0.8 GeV, for the source restricted to r <= 50, 100, 300, 1000 fm and for the full source; and a
   fit of the full source with a free intercept, C - 1 = Lambda exp(-(Q R)^alpha), over Q > 5 MeV and C - 1 > 0.1.
Run time: about 3 minutes.  Usage: python tab_lambdaPF_cuts.py
"""
import numpy as np
from scipy.optimize import least_squares
from lambdaPF import delta_analytic
from lambdaPF_source import Source, side_corr, levy_fit, HBARC, TAUF, R0

QS = np.logspace(-3.8, -0.3, 350)          # GeV
CUTS = [50.0, 100.0, 300.0, 1000.0, 3e4]   # fm


def window(k):
    """Subsonic window (lower, upper edge) of lambda-PF."""
    cs = 1 / np.sqrt(k)
    lo, hi = (k - 3 - 2 * cs) / (k - 1), (k - 3 + 2 * cs) / (k - 1)
    if k > 3 + 2 * np.sqrt(2):
        lH = 1 - 3 / k
        lo = 0.5 * (lH + np.sqrt(lH**2 - 4 / k**2))
    return lo, hi


def free_fit(y, qmin=0.005, ymin=0.1):
    """Lambda, alpha, R [fm] of C - 1 = Lambda exp(-(Q R)^alpha) over Q > qmin [GeV], y > ymin."""
    sel = (QS > qmin) & (y > ymin)
    f = lambda p: p[0] * np.exp(-(QS[sel] * p[2] / HBARC)**p[1]) - y[sel]
    return least_squares(f, [0.8, 1.3, 8.0]).x


if __name__ == '__main__':
    print('1. particle number (finite for 2 beta > 2 + kappa delta)')
    for lam in (0.30, 0.40, 0.50, 0.65):
        d = delta_analytic(lam, 5)
        print(f'   kappa=5 lambda={lam:.2f}: 2 + kappa delta = {2 + 5 * d:.2f};  '
              + ';  '.join(f'beta={b}: 2 beta = {2 * b:.1f} -> N {"finite" if 2 * b > 2 + 5 * d else "infinite"}, '
                           f'alpha_inf = {8 * b - 2 * d - 3:.1f}' for b in (0.55, 0.85)))
    print('   smallest alpha_inf compatible with a finite N and a power-law tail, over the subsonic window:')
    for k in (3, 4, 5, 6, 8, 12):
        lo, hi = window(k)
        best = (np.inf, None)
        for lam in np.linspace(lo + 1e-6, hi - 1e-6, 4001):
            d = delta_analytic(lam, k)
            if np.isnan(d):
                continue
            bmin = max(1 + k * d / 2, (1 - d) / 2)
            a = 8 * bmin - (k - 3) * d - 3
            if a < best[0]:
                best = (a, lam)
        print(f'   kappa={k}: alpha_inf > {best[0]:.3f} (at lambda = {best[1]:.3f})')
    r = np.geomspace(1e3, 1e6, 200)
    for beta in (0.55, 0.85, 1.2):          # Hubble flow: dN = n (tau_0/t) d^3r, n ~ nu
        dN = (1 + r**2 / R0**2)**(-beta) * TAUF / np.sqrt(TAUF**2 + r**2) * r**3      # per d ln r
        print(f'   Hubble flow, beta={beta}: d ln(dN/d ln r)/d ln r = {np.polyfit(np.log(r), np.log(dN), 1)[0]:+.3f} '
              f'(2 - 2 beta = {2 - 2 * beta:+.3f}; N finite if negative)')

    print('2. kappa=5, beta=0.55: alpha / R [fm] at m_T = 0.2 and 0.8 GeV for r <= r_cut; full source; free intercept')
    for lam in (0.30, 0.40, 0.50, 0.65):
        src = Source(lam, 5, 0.55, 'fast')
        for mT in (0.2, 0.8):
            ys = side_corr(src, mT, QS, CUTS)
            row = '  '.join(f'{c:g}: {a:.2f}/{R:.1f}' for c, (a, R) in ((c, levy_fit(QS, ys[c])) for c in CUTS[:-1]))
            a, R = levy_fit(QS, ys[np.inf])
            L, af, Rf = free_fit(ys[np.inf])
            print(f'   lambda={lam:.2f} m_T={mT}: {row}  full: {a:.2f}/{R:.1f}  |  free intercept: '
                  f'Lambda={L:.2f}, alpha={af:.2f}, R={Rf:.1f}', flush=True)