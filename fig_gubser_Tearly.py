"""Dressed Gubser flow: temperature at early times (figure "fig:Tearly" of the paper).

Solid: isothermal dressing, T = T_f on tau_f. Dashed: isentropic dressing (nu = const), normalised to
T = T_f at the pressure maximum on tau_f. Dotted: large-r limit of the isothermal dressing,
T_f (tau_f/tau)^(1/3). Benchmark parameters q^-1 = 4.3 fm, tau_f = 8 fm, T_f = 140 MeV.
Output: figures/fig_gubser_Tearly.pdf (+ .png preview).
Usage: python fig_gubser_Tearly.py
"""
import numpy as np
import matplotlib.pyplot as plt
import style
from gubser import T_iso, T_isen, TAUF, TF

style.apply(legend_fontsize=7.5, linewidth=1.4)

r = np.linspace(1e-3, 20, 801)
taus = (0.5, 1, 2, 4, 8)
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(taus)))
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for c, tau in zip(cols, taus):
    ax.plot(r, T_iso(tau, r)[0], '-', color=c, label=f'$\\tau={tau:g}$ fm')
    ax.plot(r, T_isen(tau, r), '--', color=c, lw=1.0)
    ax.axhline(TF * (TAUF / tau)**(1 / 3), color=c, lw=0.6, ls=':')
ax.set_xlim(0, 20); ax.set_ylim(0, 1.0)
ax.set_xlabel('$r$ [fm]'); ax.set_ylabel('$T$ [GeV]')
leg = ax.legend(loc='upper right', handlelength=1.6, labelspacing=0.25)
ax.add_artist(leg)
h1, = ax.plot([], [], 'k-', lw=1.2); h2, = ax.plot([], [], 'k--', lw=1.0); h3, = ax.plot([], [], 'k:', lw=0.8)
ax.legend([h1, h2, h3], ['isothermal', 'isentropic', r'$T_f(\tau_f/\tau)^{1/3}$'], loc='upper center',
          bbox_to_anchor=(0.47, 1.0), handlelength=1.8, labelspacing=0.25)
style.save(fig, 'fig_gubser_Tearly')

# numbers quoted in the text: hot, dilute periphery at early times
for tau in (0.5, 1.0):
    print(f'tau={tau} fm: T(r=0)={T_iso(tau, np.array([1e-4]))[0][0]:.3f} GeV, '
          f'T(r=40 fm)={T_iso(tau, np.array([40.0]))[0][0]:.3f} GeV (isothermal), '
          f'{T_isen(tau, 40.0):.3f} GeV (isentropic)')
