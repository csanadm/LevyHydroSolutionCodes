"""lambda-PF: side-direction Levy fits of the source on tau_f = 8 fm (pions, T_f = 140 MeV).

  table   Table "tab:solI" of the paper: fast-tail dressing nu = (1 + r^2/R0^2)^(-beta), R0 = 5 fm, kappa = 5,
          lambda = 0.3, 0.4 (= lambda_H, Hubble flow), 0.5, 0.65, beta = 0.55 and 0.85, m_T = 0.2 and 0.8 GeV.
          The source is restricted to r <= 300 fm; the fits of the full source (r -> infinity, with the
          analytic power-law extension of S~(0)) are printed for comparison, together with the share of S~(0)
          that comes from r > 100 fm and r > 1000 fm. Also the kappa = 3, lambda = -0.55 (c = 1.55) case quoted
          in the text.
  step1   Step 1 of the lambda-PF observables: isothermal and isentropic dressings. The paper quotes the full
          source (r <= 3e4 fm); the fits for r <= 300, 3e3 and 1e4 fm show the convergence (App. D), and the
          radius beyond which half of S~(0) is emitted shows how broad the sources are (Step 1 text).
Run time: about 2 minutes (table) and 15 minutes (step1).  Usage: python tab_lambdaPF_levy.py [table] [step1]
(default: both)
"""
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from lambdaPF_source import Source, side_corr, levy_fit

QS = np.logspace(-3.8, -0.3, 350)    # GeV
RCUT, RMAX = 300.0, 3e4              # fm
MTS = (0.2, 0.8)
STEP1_CUTS = (RCUT, 3e3, 1e4, RMAX)  # fm


def fits(src, mT):
    """(alpha, R) for r <= RCUT and for the full source, and the shares of S~(0) from r > 100 fm and r > 1000 fm."""
    ys = side_corr(src, mT, QS, [100.0, RCUT, 1000.0, RMAX])
    S0 = ys['S0']
    shares = (1 - S0[100.0] / S0[np.inf], 1 - S0[1000.0] / S0[np.inf])
    return levy_fit(QS, ys[RCUT]), levy_fit(QS, ys[np.inf]), shares


def half_radius(src, mT, rmax=RMAX, nmu=160):
    """Radius beyond which half of S~(0) = int d^3r S originates."""
    x, wx = leggauss(nmu)
    r = np.geomspace(1e-2, rmax, 4000)
    S0 = np.array([(src(ri, x, mT) * wx).sum() / 2 for ri in r])
    cum = np.cumsum(r**3 * S0 * np.gradient(np.log(r)))
    return r[np.searchsorted(cum / cum[-1], 0.5)]


def fmt(pair):
    return ' / '.join(f'{x:.2f}' for x in pair)


if __name__ == '__main__':
    which = sys.argv[1:] or ['table', 'step1']
    if 'table' in which:
        print('Table tab:solI: kappa = 5, fast-tail dressing; alpha and R [fm] at m_T = 0.2 / 0.8 GeV')
        cases = [(5, lam) for lam in (0.30, 0.40, 0.50, 0.65)] + [(3, -0.55)]
        for k, lam in cases:
            for beta in (0.55, 0.85):
                if k == 3 and beta == 0.85:
                    continue
                res = [fits(Source(lam, k, beta, 'fast'), mT) for mT in MTS]
                cut = [r[0] for r in res]
                full = [r[1] for r in res]
                sh = [r[2] for r in res]
                print(f'kappa={k} lambda={lam:+.2f} c={k * (1 - lam) / 3:.2f} beta={beta:.2f}:  '
                      f'r<={RCUT:.0f} fm: alpha {fmt([a for a, _ in cut])}, R {fmt([R for _, R in cut])}   |   '
                      f'full source: alpha {fmt([a for a, _ in full])}, R {fmt([R for _, R in full])}   |   '
                      f'share of S~(0) from r>100 / r>1000 fm: ' + ', '.join(f'{a:.1%} / {b:.1%}' for a, b in sh))
    if 'step1' in which:
        print('\nStep 1: isothermal and isentropic dressings, kappa = 5; alpha and R [fm] at m_T = 0.2 / 0.8 GeV,')
        print('for the source restricted to r <= ' + ', '.join(f'{c:g}' for c in STEP1_CUTS)
              + ' fm (the paper quotes the last one)')
        for dressing in ('iso', 'isen'):
            for lam in (0.30, 0.40, 0.50, 0.65):
                src = Source(lam, 5, dressing=dressing)
                ys = [side_corr(src, mT, QS, list(STEP1_CUTS)) for mT in MTS]
                for c in STEP1_CUTS:
                    res = [levy_fit(QS, y[c]) for y in ys]
                    print(f'{dressing:4s} lambda={lam:.2f} r<={c:7.0f} fm:  alpha {fmt([a for a, _ in res])},  '
                          f'R {fmt([R for _, R in res])}')
                print(f'{dressing:4s} lambda={lam:.2f}  half of S~(0) from r > '
                      f'{fmt([half_radius(src, mT) for mT in MTS])} fm')
