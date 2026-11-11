"""Dressed Gubser flow: systematics of the Levy fit (Appendix "Numerical methods" of the paper).

Benchmark q^-1 = 4.3 fm, tau_f = 8 fm, T_f = 140 MeV, isothermal dressing, pions, side direction.
Default fit: linear regression of ln(-ln y) on ln Q over 0.1 < y < 0.9, y = C(Q) - 1, i.e. intercept fixed to 1.
Variations: fit window 0.2 < y < 0.8 and 0.05 < y < 0.95; free intercept, y = Lambda exp(-|QR|^alpha),
fitted to ln y over 0.1 < y < 0.9.
Run time: about 20 s.  Usage: python gubser_fit_systematics.py
"""
import numpy as np
from scipy.optimize import curve_fit
import gubser as G


def fit_window(Qs, y, lo, hi):
    """alpha from the fixed-intercept fit over lo < y < hi."""
    sel = (y > lo) & (y < hi)
    return np.polyfit(np.log(Qs[sel]), np.log(-np.log(y[sel])), 1)[0]


def fit_free(Qs, y, lo=0.1, hi=0.9):
    """alpha, R [fm], Lambda from a fit of ln y = ln Lambda - (Q R)^alpha over lo < y < hi."""
    sel = (y > lo) & (y < hi)
    f = lambda q, lnL, a, R: lnL - (q * R / G.HBARC)**a
    p, _ = curve_fit(f, Qs[sel], np.log(y[sel]), p0=[0.0, 1.3, 8.0], maxfev=20000)
    return p[1], p[2], np.exp(p[0])


if __name__ == '__main__':
    print('m_T [GeV]  alpha (0.1-0.9) | shift: window 0.2-0.8  0.05-0.95 | free intercept: alpha  shift   R [fm]  Lambda')
    for mT in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        a, R, Qs, y = G.levy_fit_side(mT)
        af, Rf, Lf = fit_free(Qs, y)
        print(f'   {mT:.1f}       {a:.3f}       |       {fit_window(Qs, y, 0.2, 0.8) - a:+.3f}      '
              f'{fit_window(Qs, y, 0.05, 0.95) - a:+.3f}   |      {af:.3f}  {af - a:+.3f}   {Rf:.2f}   {Lf:.3f}')
