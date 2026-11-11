"""Dressed Gubser flow: Levy fits (figure "fig:levy" of the paper), pions, side direction, T_f = 140 MeV.
(a) alpha vs m_T: isothermal dressing (q^-1 = 4.3 fm, tau_f = 6, 8, 10 fm; values rounded as in Table "tab:levy"),
    isentropic dressing (tau_f = 8 fm), and isothermal dressing with q^-1 = 10 fm (tau_f = 8 fm).
(b) R^-2 vs m_T for the isothermal cases; points = fitted values, lines = linear fits R^-2 = a (m_T + b).
Input: figs/*.npy from gubser_levy_data.py (run it first).
Output: figures/fig_gubser_levy.pdf (+ .png preview).
Usage: python fig_gubser_levy.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import style
from gubser_levy_data import MTS, TAUFS, linfit

style.apply(legend_fontsize=7.5, linewidth=1.4)

try:
    iso = np.load(os.path.join(style.DATADIR, 'glevy.npy'))             # [tau_f] x [m_T] x (alpha, R)
    isen = np.load(os.path.join(style.DATADIR, 'gisen.npy'))            # [m_T] x (alpha, R)
    q10 = np.load(os.path.join(style.DATADIR, 'gubser_q10_pions.npy'))  # [m_T] x (alpha, R)
except FileNotFoundError:
    sys.exit('figs/*.npy not found: run "python gubser_levy_data.py" first.')

cols = plt.cm.viridis([0.15, 0.45, 0.75])
C10 = '#B71E42'
xx = np.linspace(0.15, 0.85, 10)
fig, axs = plt.subplots(2, 1, figsize=(3.4, 4.3), sharex=True, gridspec_kw=dict(hspace=0.06))

for c, tf, res in zip(cols, TAUFS, iso):
    axs[0].plot(MTS, np.round(res[:, 0], 2), 'o-', color=c, ms=3.5, label=f'isothermal, $\\tau_f={tf:g}$ fm')
    a, b = linfit(MTS, res[:, 1])
    axs[1].plot(xx, a * (xx + b), color=c, lw=0.9)
    axs[1].plot(MTS, res[:, 1]**-2, 'o', color=c, ms=3.5, label=f'$\\tau_f={tf:g}$ fm')
axs[0].plot(MTS, isen[:, 0], 's--', color='0.45', ms=3.5, label='isentropic, $\\tau_f=8$ fm')

axs[0].plot(MTS, q10[:, 0], 'D-.', color=C10, ms=3.2, label='isothermal, $q^{-1}=10$ fm')
a, b = linfit(MTS, q10[:, 1])
axs[1].plot(xx, a * (xx + b), color=C10, lw=0.9, ls='-.')
axs[1].plot(MTS, q10[:, 1]**-2, 'D', color=C10, ms=3.2, label='$q^{-1}=10$ fm, $\\tau_f=8$ fm')

axs[0].axhline(2, color='k', lw=0.6, ls=':')
axs[0].set_ylabel('$\\alpha$'); axs[0].set_ylim(1.15, 2.05)
axs[0].legend(loc='lower right', ncol=1, handlelength=1.8, labelspacing=0.25)
axs[0].text(0.03, 0.93, '(a)', transform=axs[0].transAxes, va='top')
axs[1].set_ylabel('$R^{-2}$ [fm$^{-2}$]'); axs[1].set_xlabel('$m_T$ [GeV]')
axs[1].set_xlim(0.15, 0.85); axs[1].set_ylim(0, 0.16)
axs[1].legend(loc='upper left', handlelength=1.2, labelspacing=0.25, bbox_to_anchor=(0.06, 1.0))
axs[1].text(0.03, 0.93, '(b)', transform=axs[1].transAxes, va='top')
style.save(fig, 'fig_gubser_levy')
