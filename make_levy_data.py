"""Data for Fig. fig:levy: figs/glevy.npy (as `figs_talk.py glevy`: isothermal, q^-1=4.3 fm, tau_f=6,8,10 fm)
and figs/gubser_q10_pions.npy (pion part of gubser_q10.py: q^-1=10 fm, tau_f=8 fm, grid scaled with q^-1)."""
import numpy as np, gubser as G
mts = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
res = np.array([[G.levy_fit_side(m, tauf=tf)[:2] for m in mts] for tf in (6.0, 8.0, 10.0)])
np.save('figs/glevy.npy', res)
for tf, r in zip((6, 8, 10), res):
    a, b = np.polyfit(mts, r[:, 1]**-2, 1)
    print(f'tau_f={tf}: alpha', np.round(r[:, 0], 3), f' R^-2 = {a:.3f}(mT{b/a:+.3f})')
qinv = 10.0
q10 = np.array([G.levy_fit_side(m, q=1/qinv, L=40*qinv, dx=0.02*qinv)[:2] for m in mts])
np.save('figs/gubser_q10_pions.npy', q10)
a, b = np.polyfit(mts, q10[:, 1]**-2, 1)
print('q^-1=10: alpha', np.round(q10[:, 0], 3), 'R', np.round(q10[:, 1], 2), f' R^-2 = {a:.3f}(mT{b/a:+.3f})')
