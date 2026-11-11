"""Dressed Gubser flow, isothermal dressing: single-particle spectra and the q scan.

Reproduces Table "tab:gubserscan" of the paper (tau_f = 8 fm, T_f = 140 MeV, q^-1 = 4.3 ... 12 fm):
maximal and density-weighted mean transverse velocity on tau_f, and effective temperatures of pions, kaons and
protons from linear fits of ln(E dN/d^3K) over m_T - m = 0.1-1.0 GeV. A dash means that the spectrum still rises
above m_T - m = 0.1 GeV. Also prints the numbers quoted in the text: the blue-shifted high-m_T slope of the
benchmark, the position of the spectrum maxima, and, for the Levy-fit discussion, the pion alpha at
m_T = 0.2, 0.5, 0.8 GeV and the slope a of R^-2 = a (m_T + b) for each q (grid scaled with q^-1).
Run time: about half a minute.  Usage: python tab_gubser_spectra.py
"""
import numpy as np
import gubser as G

QINV = (4.3, 6.0, 8.0, 10.0, 12.0)
SPECIES = (('pi', G.M_PI), ('K', G.M_K), ('p', G.M_P))

if __name__ == '__main__':
    print('Table tab:gubserscan (tau_f = 8 fm, T_f = 140 MeV)')
    print(' q^-1   v_max  <v_perp>  T_eff(pi)  T_eff(K)  T_eff(p)   | peak (m_T-m) K, p [GeV]')
    for qinv in QINV:
        q = 1 / qinv
        vmax = q * G.TAUF / np.sqrt(1 + (q * G.TAUF)**2)
        cells, peaks = [], []
        for name, m in SPECIES:
            pk = G.spectrum_peak(m, q)
            peaks.append(pk)
            cells.append('   --   ' if pk > 0.1 else f'{G.teff(m, q):8.2f}')
        print(f'{qinv:5.1f}  {vmax:6.2f}  {G.mean_vperp(q):7.2f}  ' + '  '.join(cells) +
              f'   | {peaks[1]:.3f}, {peaks[2]:.3f}')

    vmax = G.Q * G.TAUF / np.sqrt(1 + (G.Q * G.TAUF)**2)
    print(f'\nbenchmark q^-1 = 4.3 fm: pion T_eff = {G.teff(G.M_PI):.3f} GeV; blue-shifted limit '
          f'T_f sqrt((1+v_max)/(1-v_max)) = {G.TF * np.sqrt((1 + vmax) / (1 - vmax)):.3f} GeV')

    print('\nq scan, pion Levy fits (side): alpha at m_T = 0.2/0.5/0.8 GeV and slope a of R^-2 = a (m_T + b)')
    mts = np.array([0.2, 0.5, 0.8])
    for qinv in QINV:
        q = 1 / qinv
        res = np.array([G.levy_fit_side(mT, q=q, L=40 * qinv, dx=0.02 * qinv)[:2] for mT in mts])
        a = np.polyfit(mts, res[:, 1]**-2, 1)[0]
        print(f'q^-1 = {qinv:4.1f} fm: alpha = ' + ' / '.join(f'{x:.3f}' for x in res[:, 0]) +
              f',  a = {a:.3f} fm^-2 GeV^-1')
