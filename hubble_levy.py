"""Hubble-type Levy solution (Sec. "Hubble-type solutions" of the paper): source tail and high-m_T spectrum.

Spherical case v_x = v_y = v_z: Hubble flow u = x/tau, freeze-out at tau = tau_0, nu = (1 + r^2/R^2)^(-beta),
n ∝ nu, T = T_0/nu; tau_0 = 8 fm, R = 5 fm, T_0 = 140 MeV, pions at y = 0 with K_T along x. Source per unit volume:
    S(r, mu; K) ∝ (E - k.r/t) n/n_eq(T) exp(-K.u/T),   n_eq(T) ∝ T K_2(m/T),
with mu the cosine of the angle between r and K_T, E = m_T, r/t = tanh(eta), K.u = m_T cosh(eta) - K_T mu sinh(eta),
eta = arcsinh(r/tau_0). (This is lambda-PF with lambda = lambda_H and the fast-tail dressing, cf. lambdaPF_source.py.)
Prints
  1. the local slope d ln S / d ln r in the side direction at r = 10 and 20 fm and over r = 100-300 fm
     (beta = 0.85, m_T = 0.2 and 0.8 GeV), compared with -8 beta r^2/(r^2 + R^2) and the asymptotic -8 beta;
  2. the invariant spectrum E dN/d^3K = int d^3r S: local slopes d ln / d ln m_T at m_T = 10-300 GeV for
     beta = 0.55, 0.7, 0.85, compared with -6 beta/(1 + 2 beta); for beta = 0.85 the local inverse slope
     -1/(d ln / d m_T) at m_T = 0.7 and 3 GeV, and T_eff from a fit over m_T - m = 0.1-1.0 GeV.
Run time: about 10 s.  Usage: python hubble_levy.py
"""
import numpy as np
from scipy.special import kve

TAU0, RH, T0, M = 8.0, 5.0, 0.140, 0.13957     # fm, fm, GeV, GeV


def source(r, u, mT, beta):
    """S(r, mu) with u = 1 - mu (u in [0, 2]); written so that the forward cone mu -> 1 is evaluated stably."""
    KT = np.sqrt(mT**2 - M**2)
    eta = np.arcsinh(r / TAU0)
    nu = (1 + r**2 / RH**2)**(-beta)
    T = T0 / nu
    em = np.exp(-eta)
    ch, sh = np.cosh(eta), np.sinh(eta)
    dmk = M**2 / (mT + KT)                                     # m_T - K_T
    Ku = dmk * ch + KT * em + KT * u * sh                      # m_T cosh - K_T (1-u) sinh
    flux = dmk + KT * 2 * em**2 / (1 + em**2) + KT * u * np.tanh(eta)   # m_T - K_T (1-u) tanh
    return flux * nu / (T * kve(2, M / T)) * np.exp(-(Ku - M) / T)


def spectrum(mT, beta, nr=1500, nu_=1500):
    """E dN/d^3K (arbitrary normalisation) = 2 pi int r^2 dr int_0^2 du S, on logarithmic grids in r and u."""
    lr = np.linspace(np.log(1e-2), np.log(1e7), nr)
    lu = np.linspace(np.log(1e-16), np.log(2.0), nu_)
    r, u = np.exp(lr)[:, None], np.exp(lu)[None, :]
    f = source(r, u, mT, beta) * r**3 * u
    return 2 * np.pi * np.trapezoid(np.trapezoid(f, lu, axis=1), lr)


if __name__ == '__main__':
    beta = 0.85
    print(f'1. source, side direction, beta = {beta} (asymptotic slope -8 beta = {-8 * beta:.2f})')
    for mT in (0.2, 0.8):
        for r0 in (10.0, 20.0):
            h = 1e-4
            sl = (np.log(source(r0 * (1 + h), 1.0, mT, beta)) - np.log(source(r0 * (1 - h), 1.0, mT, beta))) / (2 * h)
            print(f'   m_T = {mT} GeV, r = {r0:.0f} fm: d ln S / d ln r = {sl:+.2f}   '
                  f'(d ln nu^4 / d ln r = {-8 * beta * r0**2 / (r0**2 + RH**2):+.2f})')
        rr = np.geomspace(100, 300, 60)
        print(f'   m_T = {mT} GeV, r = 100-300 fm: {np.polyfit(np.log(rr), np.log(source(rr, 1.0, mT, beta)), 1)[0]:+.2f}')

    print('2. spectrum: local slope d ln(E dN/d^3K) / d ln m_T')
    for b in (0.55, 0.70, 0.85):
        out = []
        for mT in (10.0, 30.0, 100.0, 300.0):
            h = 0.02
            out.append((np.log(spectrum(mT * (1 + h), b)) - np.log(spectrum(mT * (1 - h), b))) / (2 * h))
        print(f'   beta = {b:.2f}: m_T = 10/30/100/300 GeV: ' + ' / '.join(f'{x:+.3f}' for x in out) +
              f'   (-6 beta/(1+2 beta) = {-6 * b / (1 + 2 * b):+.3f})')
    for mT in (0.7, 3.0):
        h = 0.01
        d = (np.log(spectrum(mT + h, beta)) - np.log(spectrum(mT - h, beta))) / (2 * h)
        print(f'   beta = {beta}: local inverse slope at m_T = {mT} GeV: {-1 / d:.2f} GeV')
    mts = M + np.linspace(0.1, 1.0, 19)
    teff = -1 / np.polyfit(mts, np.log([spectrum(x, beta) for x in mts]), 1)[0]
    print(f'   beta = {beta}: T_eff (m_T - m = 0.1-1.0 GeV) = {teff:.2f} GeV')
