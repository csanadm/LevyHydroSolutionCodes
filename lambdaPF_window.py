"""lambda-PF: the subsonic window, case (i) of the paper, and the bounds on the core acceleration.

1. Numerical edges of the window, i.e. of the lambda range where the regular branch reaches the light cone without
   becoming sonic, for 3 <= kappa <= 12 (bisection), compared with the analytic edges: the upper edge
   lambda_+ = (kappa - 3 + 2 c_s)/(kappa - 1) from s_c = 1 (Eq. window); the lower edge
   lambda_- = (kappa - 3 - 2 c_s)/(kappa - 1) for kappa <= 3 + 2 sqrt(2), and lambda_c of Eq. (lambdac) above.
   The integration runs up to eta = 9 (r/tau ~ 4000, as in the window figure). Just below lambda_c, delta is complex
   with a small imaginary part, and the branch becomes sonic only at large eta; the lower edge for
   kappa > 3 + 2 sqrt(2) is therefore also given for eta = 30, where it approaches lambda_c.
2. Analytic numbers quoted in the text: the window for kappa = 3 and 5, lambda_c vs lambda_- for kappa = 7, the
   kappa above which lambda_- > lambda_H, and the bounds on the core acceleration c = kappa (1 - lambda)/3
   (Eq. cwindow; for kappa > 3 + 2 sqrt(2) the upper bound is kappa (1 - lambda_c)/3).
Run time: about 3 minutes.  Usage: python lambdaPF_window.py
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from lambdaPF import rhs

KAPPA_STAR = 3 + 2 * np.sqrt(2)


def reaches_lightcone(lam, k, eta_max=9.0):
    """True if the regular branch reaches eta_max (r/tau = sinh(eta_max)) without hitting the sonic line."""
    if abs(lam) < 1e-12:
        return False
    G0 = np.sign(lam); P = 3 + k * (lam - 1); Gpp0 = lam * P * G0 / 3; e0 = 1e-3
    cs = 1 / np.sqrt(k)

    def ev(eta, y, lam, k):
        return abs(y[1] / (lam * y[0])) - cs * (1 - 1e-4)
    ev.terminal = True
    sol = solve_ivp(rhs, (e0, eta_max), [G0 + 0.5 * Gpp0 * e0**2, Gpp0 * e0], args=(lam, k),
                    method='DOP853', rtol=1e-10, atol=1e-12, events=ev)
    return sol.status == 0 and len(sol.t_events[0]) == 0


def lam_pm(k):
    cs = 1 / np.sqrt(k)
    return (k - 3 - 2 * cs) / (k - 1), (k - 3 + 2 * cs) / (k - 1)


def lam_c(k):
    lH = 1 - 3 / k
    return 0.5 * (lH + np.sqrt(lH**2 - 4 / k**2))


def lower_edge(k):
    return lam_pm(k)[0] if k <= KAPPA_STAR else lam_c(k)


def edge(k, a, b, eta_max=9.0, tol=1e-5):
    """Bisection for the switch of reaches_lightcone between a and b."""
    fa = reaches_lightcone(a, k, eta_max)
    if fa == reaches_lightcone(b, k, eta_max):
        return np.nan
    while b - a > tol:
        c = 0.5 * (a + b)
        if reaches_lightcone(c, k, eta_max) == fa:
            a = c
        else:
            b = c
    return 0.5 * (a + b)


if __name__ == '__main__':
    print('1. window edges: numerical vs analytic')
    print(' kappa   lower num   lower an.              diff     | lower num (eta=30)  diff   |  upper num   lambda_+    diff')
    for k in (3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 11, 12):
        lo_an, up_an = lower_edge(k), lam_pm(k)[1]
        lo = edge(k, lo_an - 0.03, lo_an + 0.03)
        up = edge(k, up_an - 0.03, up_an + 0.03)
        kind = 'lambda_-' if k <= KAPPA_STAR else 'lambda_c'
        if k > KAPPA_STAR:
            lo30 = edge(k, lo_an - 0.03, lo_an + 0.03, eta_max=30.0)
            far = f'{lo30:+.5f}       {lo30 - lo_an:+.1e}'
        else:
            far = '       --               '
        print(f'{k:5.1f}   {lo:+.5f}   {lo_an:+.5f} ({kind})  {lo - lo_an:+.1e} | {far} |  {up:+.5f}   {up_an:+.5f}   {up - up_an:+.1e}')

    print('\n2. analytic values')
    for k in (3, 5):
        lm, lp = lam_pm(k)
        print(f'kappa={k}: window {lm:.3f} < lambda < {lp:.3f}')
    print(f'kappa=7: lambda_c = {lam_c(7):.3f}, lambda_- = {lam_pm(7)[0]:.3f}')
    k9 = brentq(lambda k: lam_pm(k)[0] - (1 - 3 / k), 6, 20)
    print(f'lambda_- > lambda_H for kappa > {k9:.3f}')
    for k in (3, 5, 7, 9):
        cs = 1 / np.sqrt(k)
        cmin, cmax = 2 * k * (1 - cs) / (3 * (k - 1)), 2 * k * (1 + cs) / (3 * (k - 1))
        txt = f'kappa={k}: {cmin:.2f} < c < {cmax:.3f} (Eq. cwindow)'
        if k > KAPPA_STAR:
            txt += f';  upper bound from lambda_c: c < {k * (1 - lam_c(k)) / 3:.3f}'
        print(txt)
