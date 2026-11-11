"""Isothermally dressed Gubser flow on tau = tau_f: profiles, transverse source, Levy fits."""
import numpy as np
from scipy.special import k1e, kv, kve

HBARC = 0.1973269804  # GeV fm
M_PI = 0.13957


def D(tau, r, q):
    return 1 + 2 * q**2 * (tau**2 + r**2) + q**4 * (tau**2 - r**2)**2


def vperp(tau, r, q):
    return 2 * q**2 * tau * r / (1 + q**2 * tau**2 + q**2 * r**2)


def pressure(tau, r, q):
    return tau**(-4 / 3) * D(tau, r, q)**(-4 / 3)


def accel(tau, r, q):
    """outward proper acceleration magnitude a = -(sinh chi d_tau + cosh chi d_r) ln w, w = p^(1/4)"""
    Dv = D(tau, r, q)
    Dt = 4 * q**2 * tau + 4 * q**4 * (tau**2 - r**2) * tau
    Dr = 4 * q**2 * r - 4 * q**4 * (tau**2 - r**2) * r
    dlnw_t = -1 / (3 * tau) - Dt / (3 * Dv)
    dlnw_r = -Dr / (3 * Dv)
    chi = np.arctanh(vperp(tau, r, q))
    return -(np.sinh(chi) * dlnw_t + np.cosh(chi) * dlnw_r)


def source_xy(x, y, mT, q, tauf, Tf, m=M_PI, dressing='iso'):
    """Transverse source at y=0 rapidity, K_T along +x (out); x,y in fm, energies in GeV."""
    KT = np.sqrt(mT**2 - m**2)
    r = np.hypot(x, y)
    v = vperp(tauf, r, q)
    chi = np.arctanh(v)
    p = pressure(tauf, r, q)
    if dressing == 'iso':
        n = p
        T = Tf * np.ones_like(r)
    elif dressing in ('isen', 'hom'):  # isentropic dressing: nu = const, T = p^(1/4)/nu, T = Tf at the pressure maximum
        rstar = np.sqrt(max(tauf**2 - q**-2, 0))
        T = Tf * (p / pressure(tauf, rstar, q))**0.25
        n = p / T
    else:
        raise ValueError(dressing)
    cphi = np.where(r > 0, x / np.where(r > 0, r, 1), 1.0)
    arg = mT * np.cosh(chi) / T
    # K1(arg) = k1e(arg) e^{-arg}; combine exponents for stability
    expo = -arg + KT * np.sinh(chi) * cphi / T
    if dressing == 'iso':          # n_eq(T_f) is a constant and drops out
        return n * k1e(arg) * np.exp(expo)
    # general dressing: fugacity n/n_eq(T), n_eq ∝ T K2(m/T) = T kve(2, m/T) e^{-m/T}  (review item 3, v8)
    return n / (T * kve(2, m / T)) * k1e(arg) * np.exp(expo + m / T)


def levy_fit_side(mT, q=1 / 4.3, tauf=8.0, Tf=0.140, dressing='iso', L=160.0, dx=0.08, m=M_PI):
    xs = np.arange(-L, L + dx / 2, dx)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    S = source_xy(X, Y, mT, q, tauf, Tf, m=m, dressing=dressing)
    P = S.sum(axis=0) * dx            # projection onto side (y) axis
    Qs = np.logspace(-3.3, -0.3, 600)  # GeV
    Ft = np.array([np.sum(P * np.cos(Q * xs / HBARC)) for Q in Qs]) * dx
    yv = (Ft / (P.sum() * dx))**2      # C(Q) - 1 = |S~(Q)|^2 / |S~(0)|^2
    sel = (yv > 0.1) & (yv < 0.9)
    A = np.polyfit(np.log(Qs[sel]), np.log(-np.log(yv[sel])), 1)
    alpha = A[0]
    R = HBARC * np.exp(A[1] / alpha)
    return alpha, R, Qs, yv


if __name__ == '__main__':
    q = 1 / 4.3
    print('r*', np.sqrt(64 - q**-2), 'vmax', q * 8 / np.sqrt(1 + (q * 8)**2))
    for r in [50, 100, 300, 1000]:
        print('r a ->', r, r * accel(8.0, r, q), 'slope p', np.log(pressure(8, r * 1.01, q) / pressure(8, r, q)) / np.log(1.01))
    for dr in ('iso', 'isen'):
        for mT in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
            a, R, _, _ = levy_fit_side(mT, dressing=dr)
            print(f'{dr} tau_f=8 mT={mT}: alpha={a:.3f} R={R:.3f} fm R^-2={R**-2:.4f}')
