"""lambda-PF with the fast-tail dressing: numerical check of the asymptotic source tail.

Prediction (Eq. alphainfI of the paper):  S ~ r^-[8 beta - (kappa-3) delta]  for 2 beta + delta >= 1,
with delta from Eq. (lightcone). At finite r (Eq. localslope):
    d ln S / d ln r ~= -[8 beta - (kappa-3) delta] + omega K.u/T,   omega = 2 beta + delta - 1.
Prints, for pions with m_T = 0.2 and 0.8 GeV, the local slopes in the side direction over r = 10^2-3x10^2,
10^3-3x10^3 and 10^4-3x10^4 fm, the finite-r prediction at the window centres, and the angle-averaged slope
at 10^2-3x10^2 fm. The kappa = 5 rows give the numbers in parentheses in Table "tab:solIstable"
(-slope - 3 at 10^4-3x10^4 fm, m_T = 0.2 GeV); lambda = lambda_H gives the Hubble-type values of Sec. IV
(-6.74 and -6.58 at 100-300 fm for beta = 0.85).
Run time: a few seconds.  Usage: python tab_lambdaPF_tails.py
"""
import numpy as np
from lambdaPF import delta_analytic
from lambdaPF_source import Source, local_slope

WINDOWS = ((100, 300), (1e3, 3e3), (1e4, 3e4))
CASES = ((5, 0.30), (5, 0.40), (5, 0.50), (5, 0.65), (3, -0.55), (3, 0.0), (3, 0.30))

if __name__ == '__main__':
    print('kappa lambda   c    delta  beta  omega  pred   | m_T | side slope 1e2-3e2 / 1e3-3e3 / 1e4-3e4 '
          '| pred + omega K.u/T at 1.7e2 / 1.7e4 | avg 1e2-3e2')
    for k, lam in CASES:
        d = delta_analytic(lam, k) if lam != 0 else 0.0
        for beta in (0.55, 0.85):
            src = Source(lam, k, beta, 'fast')
            pred = -(8 * beta - (k - 3) * d)
            om = 2 * beta + d - 1
            for mT in (0.2, 0.8):
                sd = [local_slope(src, mT, w) for w in WINDOWS]
                corr = [pred + om * float(src.KuT_side(np.array([np.sqrt(a * b)]), mT)[0]) for a, b in WINDOWS[::2]]
                av = local_slope(src, mT, WINDOWS[0], 'avg')
                print(f'  {k}  {lam:+.2f}  {k * (1 - lam) / 3:5.2f}  {d:+.3f}  {beta:.2f}  {om:+.2f}  {pred:+.3f} | '
                      f'{mT:.1f} | {sd[0]:+.3f} / {sd[1]:+.3f} / {sd[2]:+.3f} | {corr[0]:+.3f} / {corr[1]:+.3f} | {av:+.3f}')
