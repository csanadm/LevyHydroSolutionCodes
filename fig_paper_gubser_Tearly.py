"""Fig. fig:Tearly of the accelerating-hydro draft (v4): temperature of the dressed Gubser flow at early times.
Solid: isothermal dressing, Eq. (Tiso), T = T_f on tau_f. Dashed: isentropic dressing (nu = const), normalised
to T = T_f at the pressure maximum on tau_f. Dotted: the large-r limit of the isothermal dressing,
T_f (tau_f/tau)^(1/3). Parameters: q^-1 = 4.3 fm, tau_f = 8 fm, T_f = 140 MeV.
Output: figures/fig_gubser_Tearly.pdf (+ .png preview).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gubser as G
from gubser_obs import T_iso, Q, TAUF, TF

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False, 'font.size': 9,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'legend.frameon': False, 'legend.fontsize': 7.5, 'lines.linewidth': 1.4})
HERE = os.path.dirname(os.path.abspath(__file__))
r = np.linspace(1e-3, 20, 801)
taus = (0.5, 1, 2, 4, 8)
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(taus)))
rstar = np.sqrt(TAUF**2 - Q**-2)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for c, tau in zip(cols, taus):
    T = T_iso(tau, r)[0]
    Tis = TF * (G.pressure(tau, r, Q) / G.pressure(TAUF, rstar, Q))**0.25
    ax.plot(r, T, '-', color=c, label=f'$\\tau={tau:g}$ fm')
    ax.plot(r, Tis, '--', color=c, lw=1.0)
    ax.axhline(TF * (TAUF / tau)**(1 / 3), color=c, lw=0.6, ls=':')
ax.set_xlim(0, 20); ax.set_ylim(0, 1.0)
ax.set_xlabel('$r$ [fm]'); ax.set_ylabel('$T$ [GeV]')
leg = ax.legend(loc='upper right', handlelength=1.6, labelspacing=0.25)
ax.add_artist(leg)
h1, = ax.plot([], [], 'k-', lw=1.2); h2, = ax.plot([], [], 'k--', lw=1.0); h3, = ax.plot([], [], 'k:', lw=0.8)
ax.legend([h1, h2, h3], ['isothermal', 'isentropic', r'$T_f(\tau_f/\tau)^{1/3}$'], loc='upper center',
          bbox_to_anchor=(0.47, 1.0), handlelength=1.8, labelspacing=0.25)
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
fig.savefig(os.path.join(HERE, 'figures', 'fig_gubser_Tearly.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'figures', 'fig_gubser_Tearly.png'), dpi=220, bbox_inches='tight')
for tau in (0.5, 1.0):
    print(f'tau={tau}: T(0)={T_iso(tau, np.array([1e-4]))[0][0]:.3f}  T(40 fm)={T_iso(tau, np.array([40.0]))[0][0]:.3f}  '
          f'isentropic T(40 fm)={TF*(G.pressure(tau, 40.0, Q)/G.pressure(TAUF, rstar, Q))**0.25:.3f}')
