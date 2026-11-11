"""Levy fits of the dressed Gubser flow (pions, side direction, T_f = 140 MeV) for Table "tab:levy"
and figure "fig:levy" of the paper. Writes to figs/:
  glevy.npy             isothermal dressing, q^-1 = 4.3 fm, tau_f = 6, 8, 10 fm;  shape (3, 6, 2)
  gisen.npy             isentropic dressing, q^-1 = 4.3 fm, tau_f = 8 fm;          shape (6, 2)
  gubser_q10_pions.npy  isothermal dressing, q^-1 = 10 fm, tau_f = 8 fm;           shape (6, 2)
                        (transverse grid scaled with q^-1: half-width 40 q^-1, spacing 0.02 q^-1)
The last axis is (alpha, R [fm]); m_T = 0.2, 0.3, 0.4, 0.5, 0.6, 0.8 GeV.
Run time: about 1-2 minutes.  Usage: python gubser_levy_data.py
"""
import numpy as np
import gubser as G
from style import data_path

MTS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
TAUFS = (6.0, 8.0, 10.0)


def linfit(mts, R):
    """Linear fit R^-2 = a (m_T + b); returns a, b."""
    a, ab = np.polyfit(mts, R**-2, 1)
    return a, ab / a


if __name__ == '__main__':
    iso = np.array([[G.levy_fit_side(m, tauf=tf)[:2] for m in MTS] for tf in TAUFS])
    np.save(data_path('glevy.npy'), iso)
    for tf, res in zip(TAUFS, iso):
        a, b = linfit(MTS, res[:, 1])
        print(f'isothermal, tau_f={tf:g} fm: alpha', np.round(res[:, 0], 3), f' R^-2 = {a:.3f} (m_T {b:+.3f} GeV)')

    isen = np.array([G.levy_fit_side(m, tauf=8.0, dressing='isen')[:2] for m in MTS])
    np.save(data_path('gisen.npy'), isen)
    a, b = linfit(MTS, isen[:, 1])
    print('isentropic, tau_f=8 fm: alpha', np.round(isen[:, 0], 3), f' R^-2 = {a:.3f} (m_T {b:+.3f} GeV)')

    qinv = 10.0
    q10 = np.array([G.levy_fit_side(m, q=1 / qinv, L=40 * qinv, dx=0.02 * qinv)[:2] for m in MTS])
    np.save(data_path('gubser_q10_pions.npy'), q10)
    a, b = linfit(MTS, q10[:, 1])
    print('isothermal, q^-1=10 fm, tau_f=8 fm: alpha', np.round(q10[:, 0], 3), 'R', np.round(q10[:, 1], 2),
          f' R^-2 = {a:.3f} (m_T {b:+.3f} GeV)')
