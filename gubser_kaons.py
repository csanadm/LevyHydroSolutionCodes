"""Dressed Gubser flow, isothermal dressing: kaon vs pion Levy fits at equal m_T, and pion fits above m_T = 0.8 GeV.

Reproduces the numbers of the last two items of the Levy-fit list of the paper (side direction, T_f = 140 MeV,
tau_f = 8 fm): benchmark q^-1 = 4.3 fm at m_T = 0.6-1.5 GeV, and q^-1 = 10 fm (grid scaled with q^-1)
at m_T = 0.6-1.0 GeV.
Run time: about one minute.  Usage: python gubser_kaons.py
"""
import gubser as G


def compare(mT, **kw):
    ap, Rp = G.levy_fit_side(mT, m=G.M_PI, **kw)[:2]
    ak, Rk = G.levy_fit_side(mT, m=G.M_K, **kw)[:2]
    print(f'  m_T = {mT:.1f} GeV:  pi alpha = {ap:.3f}, R = {Rp:.2f} fm  |  K alpha = {ak:.3f}, R = {Rk:.2f} fm'
          f'  |  alpha_K - alpha_pi = {ak - ap:+.3f}, R_K/R_pi - 1 = {Rk / Rp - 1:+.1%}')


if __name__ == '__main__':
    print('benchmark q^-1 = 4.3 fm')
    for mT in (0.6, 0.8, 1.0, 1.2, 1.5):
        compare(mT)
    qinv = 10.0
    print('q^-1 = 10 fm')
    for mT in (0.6, 0.8, 1.0):
        compare(mT, q=1 / qinv, L=40 * qinv, dx=0.02 * qinv)
