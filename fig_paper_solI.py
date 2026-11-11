"""Paper figures for lambda-PF (Sec. VII), made with the solver solI.py:
  figures/fig_solI_profiles.pdf  (Fig. fig:solI)      regular branch for kappa = 3 and 5, panels (a)-(h)
  figures/fig_solI_window.pdf    (Fig. fig:solIwindow) fate of the regular branch in the (kappa, lambda) plane
  figures/fig_solI_accel.pdf     (Fig. fig:solIaccel)  proper acceleration on tau = const
The window scan is the one of figs_solI_reproduction.py (kappa = 3..9 in 49 steps, d lambda = 0.0125, ~2 min);
the other two figures use the same lambda values as the earlier versions of the paper figures.
Usage: python fig_paper_solI.py [profiles] [window] [accel]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, ListedColormap
from scipy.integrate import solve_ivp
from solI import solve, rhs, delta_analytic

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False, 'font.size': 9,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'legend.frameon': False, 'legend.fontsize': 6.5, 'lines.linewidth': 1.2})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures'); os.makedirs(OUT, exist_ok=True)
which = sys.argv[1:] or ['profiles', 'window', 'accel']


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=220, bbox_inches='tight')


# ------------------------------------------------------------------ Fig. fig:solI
if 'profiles' in which:
    SETS = {3: [-0.5, -0.3, 0.3, 0.5, 0.7], 5: [0.2, 0.3, 0.35, 0.5, 0.65]}
    LAMH = {3: 0.0, 5: 0.4}
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.5), gridspec_kw=dict(hspace=0.45, wspace=0.55))
    for row, k in enumerate((3, 5)):
        ax = axs[row]
        cs = 1 / np.sqrt(k)
        cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(SETS[k])))
        # Hubble flow: v = r/t, w = const on tau = const (for kappa = 3 outside the power ansatz: shown in (b), (d) only)
        xx = np.linspace(0, 1, 50)
        ax[1].plot(xx, xx, 'k--', lw=0.9)
        ax[3].plot([1e-2, 1e4], [1, 1], 'k--', lw=0.9, label=f'Hubble ($\\lambda_{{\\rm H}}={LAMH[k]:g}$)')
        if k == 5:
            sH = solve(LAMH[k], k)
            ax[0].plot(sH['eta'], sH['F'], 'k--', lw=0.9)
            ax[2].plot(sH['xi'], sH['vrel'], 'k--', lw=0.9)
        for c, lam in zip(cols, SETS[k]):
            s = solve(lam, k)
            lab = f'$\\lambda={lam:g}$' + (' (sonic)' if s['sonic'] else '')
            ax[0].plot(s['eta'], s['F'], color=c)
            ax[1].plot(s['xi'], s['v'], color=c)
            ax[2].plot(s['xi'], s['vrel'], color=c)
            ax[3].plot(s['r_over_tau'], s['w_norm'], color=c, label=lab)
            if s['sonic']:
                for a, x, y in ((ax[0], s['eta'], s['F']), (ax[1], s['xi'], s['v']), (ax[2], s['xi'], s['vrel']),
                                (ax[3], s['r_over_tau'], s['w_norm'])):
                    a.plot(x[-1], y[-1], 'x', color=c, ms=5, mew=1.3)
            else:
                d = delta_analytic(lam, k)
                xr = np.logspace(1.5, 4, 20)
                ax[3].plot(xr, s['w_norm'][-1] * (xr / s['r_over_tau'][-1])**d, ':', color='k', lw=0.8)
        ax[0].set_yscale('log'); ax[0].set_xlim(0, 10)
        ax[0].set_xlabel('$\\eta={\\rm artanh}(r/t)$'); ax[0].set_ylabel('$F/F(0)$')
        ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1); ax[1].set_xlabel('$r/t$'); ax[1].set_ylabel('$v$')
        ax[1].text(0.06, 0.9, f'$\\kappa={k}$', transform=ax[1].transAxes, va='top')
        ax[2].axhline(cs, color='k', ls='-.', lw=0.8)
        ax[2].text(0.03, cs + 0.02, '$c_s$', fontsize=8)
        ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 0.8); ax[2].set_xlabel('$r/t$'); ax[2].set_ylabel('$v_{\\rm rel}$')
        ax[3].set_xscale('log'); ax[3].set_yscale('log'); ax[3].set_xlim(1e-2, 1e4)
        ax[3].set_xlabel('$r/\\tau$'); ax[3].set_ylabel('$w(\\tau,r)/w(\\tau,0)$')
        ax[3].legend(loc='upper left', handlelength=1.5, labelspacing=0.15)
        for i, a in enumerate(ax):
            a.set_title(f'({"abcdefgh"[4*row+i]})', loc='right', fontsize=8, pad=2)
        ax[3].set_xticks([1e-2, 1e0, 1e2, 1e4])
    save(fig, 'fig_solI_profiles')

# ------------------------------------------------------------------ Fig. fig:solIwindow
if 'window' in which:
    def reaches_lightcone(lam, k, eta_max=9.0):
        if abs(lam) < 1e-9:
            return False
        G0 = np.sign(lam); p = 3 + k * (lam - 1); Gpp0 = lam * p * G0 / 3; e0 = 1e-3
        cs = 1 / np.sqrt(k)
        def ev(eta, y, lam, k):
            return abs(y[1] / (lam * y[0])) - cs * (1 - 1e-4)
        ev.terminal = True
        sol = solve_ivp(rhs, (e0, eta_max), [G0 + 0.5 * Gpp0 * e0**2, Gpp0 * e0], args=(lam, k),
                        method='DOP853', rtol=1e-9, atol=1e-11, events=ev)
        return len(sol.t_events[0]) == 0

    ks = np.linspace(3, 9, 49)
    lams = np.linspace(-1, 1, 161)
    Z = np.full((len(lams), len(ks)), np.nan)
    for j, kk in enumerate(ks):
        for i, ll in enumerate(lams):
            if reaches_lightcone(ll, kk):
                Z[i, j] = (kk + 1) * delta_analytic(ll, kk)
    np.save(os.path.join(OUT, 'window_Z.npy'), Z)
    cmap = plt.cm.RdBu_r
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.pcolormesh(ks, lams, np.where(np.isnan(Z), 1, np.nan), cmap=ListedColormap(['#D4D4D4']), shading='nearest')
    pm = ax.pcolormesh(ks, lams, Z, cmap=cmap, norm=TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=2.5), shading='nearest')
    cb = fig.colorbar(pm, ax=ax, pad=0.03)
    cb.set_label('$(\\kappa+1)\\delta$ in $p\\propto(r/\\tau)^{(\\kappa+1)\\delta}$')
    kk = np.linspace(3, 9, 300)
    cs = 1 / np.sqrt(kk)
    ax.plot(kk, (kk - 3 + 2 * cs) / (kk - 1), 'k-', lw=1.0, label='$s_c=1$ (window edges)')
    ax.plot(kk, (kk - 3 - 2 * cs) / (kk - 1), 'k-', lw=1.0)
    ax.plot(kk, 1 - 3 / kk, 'k--', lw=1.0, label='Hubble, $\\lambda_{\\rm H}=1-3/\\kappa$')
    lamH = 1 - 3 / kk
    disc = lamH**2 - 4 / kk**2
    ok = disc >= 0
    ax.plot(kk[ok], (lamH[ok] + np.sqrt(disc[ok])) / 2, 'k-.', lw=0.8, label='$\\delta$ complex inside')
    ax.plot(kk[ok], (lamH[ok] - np.sqrt(disc[ok])) / 2, 'k-.', lw=0.8)
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_xlim(3, 9); ax.set_ylim(-1, 1)
    ax.set_xlabel('$\\kappa=1/c_s^2$'); ax.set_ylabel('$\\lambda$')
    ax.text(3.15, 0.92, 'sonic point', va='top', fontsize=7.5)
    ax.legend(loc='lower right', fontsize=7, handlelength=2.0, labelspacing=0.2)
    save(fig, 'fig_solI_window')

# ------------------------------------------------------------------ Fig. fig:solIaccel
if 'accel' in which:
    SETS = {3: [-0.5, -0.3, 0.3, 0.5], 5: [0.30, 0.35, 0.50, 0.65]}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, k in zip(axs, (3, 5)):
        ax.axhline(0, color='k', ls='--', lw=0.9, label='Hubble')
        cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(SETS[k])))
        for c, lam in zip(cols, SETS[k]):
            s = solve(lam, k)
            ax.plot(s['r_over_tau'], s['taua'], color=c, label=f'$\\lambda={lam:g}$ ($c={k*(1-lam)/3:.2f}$)')
        ax.set_xscale('log'); ax.set_xlim(1e-2, 1e4)
        ax.set_xlabel('$r/\\tau$'); ax.set_title(f'$\\kappa={k}$', fontsize=9)
        ax.legend(loc='upper left' if k == 5 else 'center right', handlelength=1.6, labelspacing=0.15)
    axs[0].set_ylabel('$\\tau a$ on $\\tau=$ const'); axs[0].set_ylim(-0.75, 0.85)
    save(fig, 'fig_solI_accel')
print('done')
