"""Dressed Gubser flow: flow profile, isothermal dressing, transverse source and Levy fits.

Milne coordinates tau, eta_s, transverse radius r; conformal EoS eps = 3p, p = nT.
Units: lengths in fm, energies and momenta in GeV.
The default parameters are the benchmark values of the paper:
q^-1 = 4.3 fm, tau_f = 8 fm, T_f = 140 MeV, pions.
"""
import numpy as np
from scipy.special import k1e, kve, i0e
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

HBARC = 0.1973269804   # GeV fm
M_PI, M_K, M_P = 0.13957, 0.49368, 0.93827   # GeV
Q, TAUF, TF = 1 / 4.3, 8.0, 0.140   # benchmark q [1/fm], tau_f [fm], T_f [GeV]


def D(tau, r, q):
    """D = 1 + 2 q^2 (tau^2 + r^2) + q^4 (tau^2 - r^2)^2."""
    return 1 + 2 * q**2 * (tau**2 + r**2) + q**4 * (tau**2 - r**2)**2


def vperp(tau, r, q):
    """Transverse flow velocity of Gubser flow."""
    return 2 * q**2 * tau * r / (1 + q**2 * tau**2 + q**2 * r**2)


def pressure(tau, r, q):
    """Pressure up to a constant factor, p ~ tau^(-4/3) D^(-4/3)."""
    return tau**(-4 / 3) * D(tau, r, q)**(-4 / 3)


def accel(tau, r, q):
    """Outward proper acceleration, a = -(sinh(chi) d_tau + cosh(chi) d_r) ln w with w = p^(1/4)."""
    Dv = D(tau, r, q)
    Dt = 4 * q**2 * tau + 4 * q**4 * (tau**2 - r**2) * tau
    Dr = 4 * q**2 * r - 4 * q**4 * (tau**2 - r**2) * r
    dlnw_t = -1 / (3 * tau) - Dt / (3 * Dv)
    dlnw_r = -Dr / (3 * Dv)
    chi = np.arctanh(vperp(tau, r, q))
    return -(np.sinh(chi) * dlnw_t + np.cosh(chi) * dlnw_r)


def T_iso(tau, r, q=Q, tauf=TAUF, Tf=TF):
    """Temperature of the isothermal dressing (T = T_f on tau = tau_f):
    T = T_f (tau_f/tau) [cosh rho_f(theta) / cosh rho(tau, r)]^(2/3).
    Returns T and the freeze-out radius r_f of the streamline through (tau, r)."""
    cot = (1 + q**2 * tau**2 - q**2 * r**2) / (2 * q * r)
    qrf = -cot + np.sqrt(cot**2 + 1 + q**2 * tauf**2)
    rf = qrf / q
    chf = np.sqrt(D(tauf, rf, q)) / (2 * q * tauf)
    ch = np.sqrt(D(tau, r, q)) / (2 * q * tau)
    return Tf * (tauf / tau) * (chf / ch)**(2 / 3), rf


def T_isen(tau, r, q=Q, tauf=TAUF, Tf=TF):
    """Temperature of the isentropic dressing (nu = const), T ~ p^(1/4),
    normalised to T = T_f at the pressure maximum on tau_f, r* = sqrt(tau_f^2 - q^-2)."""
    rstar = np.sqrt(max(tauf**2 - q**-2, 0))
    return Tf * (pressure(tau, r, q) / pressure(tauf, rstar, q))**0.25


def source_xy(x, y, mT, q, tauf, Tf, m=M_PI, dressing='iso'):
    """Transverse source on tau = tau_f at pair rapidity y = 0, integrated over eta_s,
    with K_T along +x (out direction); x, y in fm. Unnormalised.
    dressing = 'iso': isothermal, T = T_f; the constant n_eq(T_f) drops out.
    dressing = 'isen': isentropic, with the full fugacity n / n_eq(T), n_eq ~ T K_2(m/T)."""
    KT = np.sqrt(mT**2 - m**2)
    r = np.hypot(x, y)
    chi = np.arctanh(vperp(tauf, r, q))
    p = pressure(tauf, r, q)
    if dressing == 'iso':
        T = Tf * np.ones_like(r)
    elif dressing == 'isen':
        T = T_isen(tauf, r, q, tauf, Tf)
    else:
        raise ValueError(f"unknown dressing '{dressing}' (use 'iso' or 'isen')")
    n = p / T
    cphi = np.where(r > 0, x / np.where(r > 0, r, 1), 1.0)
    arg = mT * np.cosh(chi) / T
    # K_1(arg) = k1e(arg) e^(-arg); the exponents are combined for numerical stability
    expo = -arg + KT * np.sinh(chi) * cphi / T
    if dressing == 'iso':
        return n * k1e(arg) * np.exp(expo)
    # n_eq(T) ~ T K_2(m/T) = T kve(2, m/T) e^(-m/T)
    return n / (T * kve(2, m / T)) * k1e(arg) * np.exp(expo + m / T)


def levy_fit_side(mT, q=Q, tauf=TAUF, Tf=TF, dressing='iso', L=160.0, dx=0.08, m=M_PI):
    """Levy fit of the side-direction correlation function C(Q) - 1 = exp(-|QR|^alpha).

    The source is evaluated on a square grid of half-width L and spacing dx (fm), projected onto the
    side axis, and cosine-transformed at 600 log-spaced Q values between 0.5 MeV and 0.5 GeV.
    alpha and R follow from a linear fit of ln(-ln y) vs ln Q over 0.1 < y < 0.9 (intercept fixed to 1).
    Returns alpha, R [fm], Q [GeV], y = C(Q) - 1."""
    xs = np.arange(-L, L + dx / 2, dx)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    S = source_xy(X, Y, mT, q, tauf, Tf, m=m, dressing=dressing)
    P = S.sum(axis=0) * dx              # projection onto the side (y) axis
    Qs = np.logspace(-3.3, -0.3, 600)   # GeV
    Ft = np.array([np.sum(P * np.cos(Qv * xs / HBARC)) for Qv in Qs]) * dx
    yv = (Ft / (P.sum() * dx))**2       # C(Q) - 1 = |S~(Q)|^2 / |S~(0)|^2
    sel = (yv > 0.1) & (yv < 0.9)
    A = np.polyfit(np.log(Qs[sel]), np.log(-np.log(yv[sel])), 1)
    alpha = A[0]
    R = HBARC * np.exp(A[1] / alpha)
    return alpha, R, Qs, yv


# ------------------------------------------------------------------ single-particle spectra (isothermal dressing)
def spectrum(mT, m, q=Q, tauf=TAUF, Tf=TF, rmax=400.0):
    """E dN/d^3K at y = 0 for the isothermal dressing, up to a constant factor:
    m_T int r dr n(tau_f, r) K_1(m_T cosh(chi)/T_f) I_0(K_T sinh(chi)/T_f), with n ~ p on tau_f."""
    KT = np.sqrt(max(mT**2 - m**2, 0.0))

    def f(r):
        v = vperp(tauf, r, q); ch = 1 / np.sqrt(1 - v**2); sh = v * ch
        a, b = mT * ch / Tf, KT * sh / Tf
        return r * pressure(tauf, r, q) * k1e(a) * i0e(b) * np.exp(b - a + mT / Tf)   # rescaled by e^(m_T/T_f)
    return mT * quad(f, 0, rmax, limit=400)[0] * np.exp(-mT / Tf)


def teff(m, q=Q, tauf=TAUF, Tf=TF, window=(0.1, 1.0)):
    """Effective temperature: -1/slope of a linear fit of ln(E dN/d^3K) vs m_T over m_T - m in window (GeV)."""
    mts = m + np.linspace(*window, 19)
    return -1 / np.polyfit(mts, np.log([spectrum(x, m, q, tauf, Tf) for x in mts]), 1)[0]


def spectrum_peak(m, q=Q, tauf=TAUF, Tf=TF):
    """m_T - m (GeV) where the spectrum is maximal (0 if it falls monotonically)."""
    f = lambda x: -np.log(spectrum(m + x, m, q, tauf, Tf))
    grid = np.linspace(0, 2, 41)
    i = int(np.argmin([f(x) for x in grid]))
    if i == 0:
        return 0.0
    return minimize_scalar(f, bracket=(grid[i - 1], grid[i], grid[i + 1])).x


def mean_vperp(q=Q, tauf=TAUF, rmax=80.0):
    """Density-weighted mean transverse velocity on tau_f (isothermal dressing, n ~ p)."""
    rr = np.linspace(0, rmax, 8001)
    w = rr * pressure(tauf, rr, q)
    return np.sum(w * vperp(tauf, rr, q)) / np.sum(w)


# ------------------------------------------------------------------ Gaussian (saddle-point) homogeneity lengths
def gauss_radii(mT, q=Q, tauf=TAUF, Tf=TF, m=M_PI, exact_K1=False):
    """Saddle-point radii of the isothermal transverse source on the K_T axis (Eq. radii of the paper).
    With exact_K1=False only the exponential of K_1 is kept, K_1(A) ~ e^(-A); with exact_K1=True the full
    ln K_1 enters the saddle condition and the curvature. Returns r_s, v_perp(r_s), R_side, R_out, R_long (fm)."""
    KT = np.sqrt(mT**2 - m**2)

    def Psi(r):
        chi = np.arctanh(vperp(tauf, r, q))
        A = mT * np.cosh(chi) / Tf
        lnK1 = np.log(k1e(A)) - A if exact_K1 else -A
        return np.log(pressure(tauf, r, q)) + lnK1 + KT * np.sinh(chi) / Tf
    h = 1e-4
    rs = brentq(lambda r: (Psi(r + h) - Psi(r - h)) / (2 * h), 2, 15)
    chi = np.arctanh(vperp(tauf, rs, q))
    d2 = (Psi(rs + 1e-3) - 2 * Psi(rs) + Psi(rs - 1e-3)) / 1e-6
    R_side = np.sqrt(Tf * rs**2 / (KT * np.sinh(chi)))
    R_out = 1 / np.sqrt(-d2)
    R_long = np.sqrt(tauf**2 * Tf / (mT * np.cosh(chi)))
    return rs, np.tanh(chi), R_side, R_out, R_long


def emission_time_term(mT, q=Q, tauf=TAUF, Tf=TF, m=M_PI, L=80.0, dx=0.1):
    """Contribution of the emission time to R_out^2 for the isothermal source on tau = tau_f (y = 0, K_T along x):
    R_out^2 = Var(x - beta_T t) = Var(x) + [beta_T^2 Var(t) - 2 beta_T Cov(x, t)],  beta_T = K_T/m_T,
    with t = tau_f cosh(eta_s). The eta_s integrals are done analytically with
    int d eta cosh(n eta) e^(-A cosh eta) = 2 K_n(A); the transverse plane on a grid (half-width L, spacing dx, fm).
    Returns the bracket [fm^2], Var(t) [fm^2] and Cov(x, t) [fm^2]."""
    x = np.arange(-L, L + dx / 2, dx)
    X, Y = np.meshgrid(x, x, indexing='ij')
    r = np.hypot(X, Y)
    chi = np.arctanh(vperp(tauf, r, q))
    cphi = np.where(r > 0, X / np.where(r > 0, r, 1), 1.0)
    KT = np.sqrt(mT**2 - m**2)
    b = KT / mT
    A = mT * np.cosh(chi) / Tf
    base = pressure(tauf, r, q) * np.exp(-A + KT * np.sinh(chi) * cphi / Tf)      # kve(n, A) = K_n(A) e^A
    k0, k1, k2, k3 = (kve(n, A) for n in (0, 1, 2, 3))
    w0 = base * 2 * k1                                   # int cosh(eta) e^(-A cosh eta)
    w1 = base * tauf * (k0 + k2)                         # ... times t = tau_f cosh(eta)
    w2 = base * tauf**2 * (3 * k1 + k3) / 2              # ... times t^2
    Z = w0.sum()
    t1, t2 = w1.sum() / Z, w2.sum() / Z
    x1, xt = (w0 * X).sum() / Z, (w1 * X).sum() / Z
    var_t, cov = t2 - t1**2, xt - x1 * t1
    return b**2 * var_t - 2 * b * cov, var_t, cov


if __name__ == '__main__':
    # benchmark properties quoted in the paper, and the Levy fits for tau_f = 8 fm
    print(f'pressure ring r* = {np.sqrt(TAUF**2 - Q**-2):.2f} fm, '
          f'v_max = {Q * TAUF / np.sqrt(1 + (Q * TAUF)**2):.3f} at r = {np.sqrt(TAUF**2 + Q**-2):.2f} fm')
    for r in (50, 100, 300, 1000):
        slope = np.log(pressure(TAUF, r * 1.01, Q) / pressure(TAUF, r, Q)) / np.log(1.01)
        print(f'r = {r:5d} fm:  r a = {r * accel(TAUF, r, Q):.4f} (-> 2),  d ln p / d ln r = {slope:.4f} (-> -16/3)')
    for dr in ('iso', 'isen'):
        for mT in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
            a, R, _, _ = levy_fit_side(mT, dressing=dr)
            print(f'{dr:4s} tau_f=8 fm, m_T={mT} GeV: alpha={a:.3f}  R={R:.3f} fm  R^-2={R**-2:.4f} fm^-2')
