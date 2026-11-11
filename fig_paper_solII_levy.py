"""Fig. fig:levy of the accelerating-hydro draft (v4; the q^-1 = 10 fm curves were added in v4): Levy fits of the dressed Gubser flow.
(a) alpha(m_T): values of Table tab:levy; (b) R^-2(m_T): points from figs/glevy.npy
(run `python figs_talk.py glevy` first), lines = linear fits of Table tab:levy.
Output: figures/fig_solII_levy.pdf (+ .png preview).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False, 'font.size': 9,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'legend.frameon': False, 'legend.fontsize': 7.5, 'lines.linewidth': 1.4})
mTs = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
HERE = os.path.dirname(os.path.abspath(__file__))
res = np.load(os.path.join(HERE, 'figs', 'glevy.npy'))          # [tau_f=6,8,10] x mT x (alpha, R)
hom = np.array([1.70, 1.83, 1.88, 1.91, 1.93, 1.96])      # draft Table (isentropic dressing, tau_f = 8 fm)
cols = plt.cm.viridis([0.15, 0.45, 0.75])
fig, axs = plt.subplots(2, 1, figsize=(3.4, 4.3), sharex=True, gridspec_kw=dict(hspace=0.06))
TAB_A = {6: [1.71, 1.66, 1.66, 1.68, 1.70, 1.74], 8: [1.69, 1.67, 1.68, 1.71, 1.73, 1.78], 10: [1.68, 1.66, 1.69, 1.73, 1.75, 1.79]}
TAB_FIT = {6: (0.170, 0.069), 8: (0.148, 0.025), 10: (0.127, 0.002)}
for c, tf, r in zip(cols, (6, 8, 10), res):
    axs[0].plot(mTs, TAB_A[tf], 'o-', color=c, ms=3.5, label=f'isothermal, $\\tau_f={tf}$ fm')
    y = r[:, 1]**-2
    a, b = TAB_FIT[tf]
    xx = np.linspace(0.15, 0.85, 10)
    axs[1].plot(xx, a * (xx + b), color=c, lw=0.9)
    axs[1].plot(mTs, y, 'o', color=c, ms=3.5, label=f'$\\tau_f={tf}$ fm')
axs[0].plot(mTs, hom, 's--', color='0.45', ms=3.5, label='isentropic, $\\tau_f=8$ fm')
# realistic-flow variant q^-1 = 10 fm, tau_f = 8 fm (figs/gubser_q10_pions.npy from gubser_q10.py)
q10 = np.load(os.path.join(HERE, 'figs', 'gubser_q10_pions.npy'))
C10 = '#B71E42'
axs[0].plot(mTs, q10[:, 0], 'D-.', color=C10, ms=3.2, label='isothermal, $q^{-1}=10$ fm')
a10, b10 = np.polyfit(mTs, q10[:, 1]**-2, 1)
xx = np.linspace(0.15, 0.85, 10)
axs[1].plot(xx, a10 * xx + b10, color=C10, lw=0.9, ls='-.')
axs[1].plot(mTs, q10[:, 1]**-2, 'D', color=C10, ms=3.2, label='$q^{-1}=10$ fm, $\\tau_f=8$ fm')
axs[0].axhline(2, color='k', lw=0.6, ls=':')
axs[0].set_ylabel('$\\alpha$'); axs[0].set_ylim(1.15, 2.05)
axs[0].legend(loc='lower right', ncol=1, handlelength=1.8, labelspacing=0.25)
axs[0].text(0.03, 0.93, '(a)', transform=axs[0].transAxes, va='top')
axs[1].set_ylabel('$R^{-2}$ [fm$^{-2}$]'); axs[1].set_xlabel('$m_T$ [GeV]')
axs[1].set_xlim(0.15, 0.85); axs[1].set_ylim(0, 0.16)
axs[1].legend(loc='upper left', handlelength=1.2, labelspacing=0.25, bbox_to_anchor=(0.06, 1.0))
axs[1].text(0.03, 0.93, '(b)', transform=axs[1].transAxes, va='top')
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
fig.savefig(os.path.join(HERE, 'figures', 'fig_solII_levy.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'figures', 'fig_solII_levy.png'), dpi=220, bbox_inches='tight')
