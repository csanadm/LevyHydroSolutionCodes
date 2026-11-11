"""Dressed Gubser flow (isothermal dressing): origin of the Levy-like shape, and the out and long directions.

The transverse source on tau_f is S_perp ~ n(r) exp(-A cosh(eta_s) + b cos(phi)) cosh(eta_s), with
A = m_T cosh(chi)/T_f, b = K_T sinh(chi)/T_f. The eta_s and phi integrals of its Fourier transform are done
analytically, so that S~(Q) is a single radial integral (hbar = 1):
    side:  S~ ~ int r dr n K_1(A) I_0(sqrt(b^2 - Q^2 r^2))                       (I_0 of an imaginary argument is J_0)
    out:   S~ ~ int r dr n K_1(A + i Q beta_T tau_f) I_0(b + i Q r),             Q^0 = beta_T Q, beta_T = K_T/m_T
    long:  S~ ~ int r dr n A K_1(sqrt(A^2 + Q^2 tau_f^2))/sqrt(A^2 + Q^2 tau_f^2) I_0(b)
(midpoint rule, dr = 0.01 fm, r <= 3000 fm). This is independent of the two-dimensional grid of gubser.levy_fit_side
and reproduces its side-direction fits. The Levy fit is the same as there (0.1 < C - 1 < 0.9, intercept fixed to 1).
Prints, for q^-1 = 4.3 fm (benchmark) and 10 fm, tau_f = 8 fm, T_f = 140 MeV, pions, m_T = 0.2, 0.5, 0.8 GeV:
  1. alpha for the source restricted to r <= r_cut, and the share of the emission from r > r_cut (table "gubsercut");
  2. alpha and R in the side, out and long directions (table "gubserdir");
  3. density relative to its maximum, v_perp and the local exponent of the pressure at r = 20, 30, 50, 100 fm.
Run time: about 5 minutes.  Usage: python gubser_tail.py
"""
import numpy as np
from scipy.special import k1e, i0e, j0, kve, ive
from gubser import HBARC, M_PI, TAUF, TF, pressure, vperp

RMAX, DR = 3000.0, 0.01
QS = np.logspace(-3.3, -0.3, 300)        # GeV
CUTS = (10.0, 15.0, 20.0, 30.0, 50.0, 100.0, np.inf)
MTS = (0.2, 0.5, 0.8)


def profile(q, mT, tauf=TAUF, Tf=TF, m=M_PI):
    """r grid, weights r n dr, A = m_T cosh(chi)/T_f, b = K_T sinh(chi)/T_f, beta_T."""
    r = np.arange(DR / 2, RMAX, DR)
    v = vperp(tauf, r, q)
    ch = 1 / np.sqrt(1 - v**2)
    KT = np.sqrt(mT**2 - m**2)
    return r, r * pressure(tauf, r, q) * DR, mT * ch / Tf, KT * v * ch / Tf, KT / mT


def ft_side(q, mT, cuts=(np.inf,)):
    """S~(Q) along the side direction for the source restricted to r <= cut; first element: Q = 0."""
    r, w, A, b, _ = profile(q, mT)
    out = {c: [] for c in cuts}
    for Q in np.r_[0.0, QS]:
        z2 = b**2 - (Q * r / HBARC)**2
        z = np.sqrt(np.abs(z2))
        f = np.where(z2 >= 0, k1e(A) * i0e(z) * np.exp(np.minimum(z - A, 0.0)), k1e(A) * np.exp(-A) * j0(z))
        cum = np.cumsum(w * f)
        for c in cuts:
            out[c].append(cum[-1] if np.isinf(c) else cum[np.searchsorted(r, c) - 1])
    return {c: np.array(x) for c, x in out.items()}


def ft_out(q, mT, tauf=TAUF):
    r, w, A, b, bT = profile(q, mT)
    res = []
    for Q in np.r_[0.0, QS]:
        zA, zb = A + 1j * Q * bT * tauf / HBARC, b + 1j * Q * r / HBARC
        res.append(np.sum(w * kve(1, zA) * np.exp(-zA) * ive(0, zb) * np.exp(zb.real)))     # K_1(zA) I_0(zb)
    return np.array(res)


def ft_long(q, mT, tauf=TAUF):
    r, w, A, b, _ = profile(q, mT)
    res = []
    for Q in np.r_[0.0, QS]:
        s = np.sqrt(A**2 + (Q * tauf / HBARC)**2)
        res.append(np.sum(w * A * k1e(s) / s * i0e(b) * np.exp(b - s)))
    return np.array(res)


def fit(St):
    """alpha, R [fm] from y = |S~(Q)|^2/|S~(0)|^2 over 0.1 < y < 0.9 (first passage through the window)."""
    y = np.abs(St[1:])**2 / np.abs(St[0])**2
    end = np.argmax(y < 0.1) if np.any(y < 0.1) else len(y)
    Q, y = QS[:end], y[:end]
    sel = (y > 0.1) & (y < 0.9)
    a = np.polyfit(np.log(Q[sel]), np.log(-np.log(y[sel])), 1)
    return a[0], HBARC * np.exp(a[1] / a[0])


if __name__ == '__main__':
    for qinv in (4.3, 10.0):
        q = 1 / qinv
        print(f'q^-1 = {qinv} fm')
        print('1. source restricted to r <= r_cut: alpha at m_T = ' + '/'.join(str(x) for x in MTS)
              + ' GeV;  share of the emission from r > r_cut')
        St = {mT: ft_side(q, mT, CUTS) for mT in MTS}
        for c in CUTS:
            al = [fit(St[mT][c])[0] for mT in MTS]
            sh = [100 * (1 - St[mT][c][0] / St[mT][np.inf][0]) for mT in MTS]
            print(f'   r_cut = {c:5g} fm:  alpha = ' + ' / '.join(f'{x:.2f}' for x in al)
                  + '   share = ' + ' / '.join(f'{x:.1f}%' for x in sh))
        print('2. directions (full source): alpha, R [fm]')
        for mT in MTS:
            res = [fit(St[mT][np.inf]), fit(ft_out(q, mT)), fit(ft_long(q, mT))]
            print(f'   m_T = {mT} GeV:  ' + '   '.join(f'{nm} {a:.2f}, {R:.2f}'
                                                        for nm, (a, R) in zip(('side', 'out', 'long'), res)))
        print('3. outer region on tau_f')
        rr = np.linspace(0, 30, 30001)
        pmax = pressure(TAUF, rr, q).max()
        for r0 in (15.0, 20.0, 30.0, 50.0, 100.0):
            sl = np.log(pressure(TAUF, r0 * 1.001, q) / pressure(TAUF, r0 / 1.001, q)) / (2 * np.log(1.001))
            print(f'   r = {r0:5.0f} fm:  n/n_max = {pressure(TAUF, r0, q) / pmax:.2e},  v_perp = {vperp(TAUF, r0, q):.2f},'
                  f'  d ln p / d ln r = {sl:.2f}')