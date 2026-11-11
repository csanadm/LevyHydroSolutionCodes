"""lambda-PF: source function on tau = tau_f and side-direction Levy fits.

Flow. The regular branch is integrated with the first-order form of the field equation (Eq. ueq of the paper),
    u = G'/(lambda G) = -tanh(zeta),   u' = (1 - u^2) N / (1 - kappa u^2),
    N = kappa (lambda - lambda_H) - 2 u coth(eta) - kappa lambda u^2,   (ln G)' = lambda u,
with w/w(0) = (G/G_0) sqrt(1 - u^2) and flow rapidity eta + zeta. This is independent of the second-order
solver in lambdaPF.py and serves as a cross-check of it.

Source. Cooper-Frye on tau = tau_f per unit volume, pions at y = 0, K_T along the out direction:
    S(r, mu; K) ∝ (E - k.r/t) n/n_eq(T) exp(-K.u/T),   n_eq(T) ∝ T K_2(m/T)   (full K_2, no T >> m approximation),
mu = cosine of the angle between r and K_T. Dressings:
    'fast'  nu = (1 + r^2/R0^2)^(-beta) on tau_f:  n = w^kappa nu,  T = T_f (w/w0)/nu   (Hubble-type, fast tail)
    'iso'   T = T_f,  n ∝ w^(kappa+1)
    'isen'  T = T_f w/w0,  n ∝ w^kappa
Parameters as in the paper: tau_f = 8 fm, R0 = 5 fm, T_f = 140 MeV.

Correlation function. S~(Q) for Q along the side direction from a Legendre expansion of S in mu (l <= L, even l
only) and spherical Bessel functions; cumulative radial integrals give S~ for several cuts r <= r_cut at once.
Levy fit: linear regression of ln(-ln y) on ln Q over 0.1 < y < 0.9, y = C(Q) - 1 (intercept fixed to 1).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import kve, spherical_jn, gammaln
from numpy.polynomial.legendre import leggauss, legval

HBARC = 0.1973269804                 # GeV fm
M_PI = 0.13957                       # GeV
TAUF, R0, TF = 8.0, 5.0, 0.140       # fm, fm, GeV


def flow(lam, k, eta_max=9.2):
    """Regular branch on an eta grid: returns eta, ln(w/w0), flow rapidity Y = eta + zeta."""
    if k == 3 and lam == 0:          # kappa = 3 Hubble flow (Phi = ln tau, outside the power ansatz)
        eta = np.linspace(0, eta_max, 2000)
        return eta, np.zeros_like(eta), eta.copy()
    P = 3 + k * (lam - 1)

    def f(e, y):
        u = y[0]
        N = P - 2 * u / np.tanh(e) - k * lam * u**2
        return [(1 - u**2) * N / (1 - k * u**2), lam * u]      # y = (u, ln G - ln G0)
    e0 = 1e-5
    eta = np.r_[0, np.geomspace(e0, eta_max, 20000)]
    sol = solve_ivp(f, (e0, eta_max), [P * e0 / 3, lam * P * e0**2 / 6], t_eval=eta[1:],
                    method='DOP853', rtol=1e-12, atol=1e-14)
    if sol.status != 0 or np.any(k * sol.y[0]**2 >= 1):
        raise ValueError(f'kappa={k}, lambda={lam}: the regular branch reaches the sonic line (outside the window)')
    u = np.r_[0, sol.y[0]]
    lnG = np.r_[0, sol.y[1]]
    return eta, lnG + 0.5 * np.log1p(-u**2), eta - np.arctanh(u)


class Source:
    """Source function S(r, mu, m_T) on tau_f for given lambda, kappa, dressing (and beta for 'fast')."""

    def __init__(self, lam, k, beta=None, dressing='fast'):
        if dressing not in ('fast', 'iso', 'isen'):
            raise ValueError(dressing)
        self.eta, self.lnw, self.Y = flow(lam, k)
        self.lam, self.k, self.beta, self.dressing = lam, k, beta, dressing

    def thermo(self, r):
        """w/w0, flow rapidity Y, T and n at radius r on tau_f."""
        e = np.arcsinh(r / TAUF)
        w = np.exp(np.interp(e, self.eta, self.lnw))
        Y = np.interp(e, self.eta, self.Y)
        if self.dressing == 'fast':
            nu = (1 + r**2 / R0**2)**(-self.beta)
            T, n = TF * w / nu, w**self.k * nu
        elif self.dressing == 'iso':
            T, n = TF * np.ones_like(w), w**(self.k + 1)
        else:
            T, n = TF * w, w**self.k
        return e, w, Y, T, n

    def __call__(self, r, mu, mT, m=M_PI):
        KT = np.sqrt(mT**2 - m**2)
        e, w, Y, T, n = self.thermo(r)
        flux = mT - KT * mu * np.tanh(e)                    # E - k.r/t with E = m_T at y = 0, r/t = tanh(eta)
        Ku = mT * np.cosh(Y) - KT * mu * np.sinh(Y)
        # n/n_eq(T) exp(-K.u/T) with n_eq ∝ T K_2(m/T) = T kve(2, m/T) e^(-m/T)
        return flux * n / (T * kve(2, m / T)) * np.exp(-(Ku - m) / T)

    def KuT_side(self, r, mT):
        """K.u/T in the side direction (mu = 0)."""
        _, _, Y, T, _ = self.thermo(r)
        return mT * np.cosh(Y) / T


def local_slope(src, mT, window, mode='side', npts=60, nmu=64):
    """d ln S / d ln r from a linear fit of ln S vs ln r over window = (r1, r2) fm, in the side direction
    (mu = 0) or for the angle-averaged source ('avg')."""
    r = np.geomspace(*window, npts)
    if mode == 'side':
        S = src(r, 0.0, mT)
    else:
        x, wx = leggauss(nmu)
        S = sum(wi * src(r, xi, mT) for xi, wi in zip(x, wx)) / 2
    return np.polyfit(np.log(r), np.log(S), 1)[0]


def _rgrid(rmax, per_efold=1500):
    """Radial grid (linear up to 1 fm, then logarithmic) with trapezoid weights."""
    a = np.arange(0, 1.0, 1 / per_efold)
    b = np.exp(np.arange(0, np.log(rmax), 1 / per_efold))
    r = np.r_[a, b, rmax]
    w = np.zeros_like(r)
    d = np.diff(r)
    w[:-1] += d / 2
    w[1:] += d / 2
    return r, w


def side_corr(src, mT, Qs, cuts, L=60, nmu=160, Nosc=400.0, m=M_PI):
    """y(Q) = |S~(Q)|^2 / |S~(0)|^2 for Q along the side direction, for the source restricted to r <= cut,
    for each cut in cuts (fm). Returns a dict cut -> y; the key np.inf holds the result with the analytic
    power-law extension of S~(0) beyond max(cuts), 'tailslope' the local power of S_0(r) used for it, and 'S0' a dict
    cut -> S~(0) (unnormalised; np.inf: with the extension)."""
    r, wr = _rgrid(max(cuts))
    x, wx = leggauss(nmu)
    S = src(r[:, None], x[None, :], mT, m)                              # (N_r, n_mu)
    ls = np.arange(0, L + 1, 2)
    Pl = np.array([legval(x, np.eye(L + 1)[l]) for l in ls])            # (n_l, n_mu)
    Sl = (S * wx) @ Pl.T                                                # S_l(r), (N_r, n_l)
    # (2l+1) P_l(0) i^l = (2l+1) (l-1)!!/l!! for even l;  (l-1)!!/l!! = l! / (2^l ((l/2)!)^2)
    coef = np.exp(gammaln(ls + 1) - ls * np.log(2) - 2 * gammaln(ls / 2 + 1)) * (2 * ls + 1)
    c0 = np.cumsum(r**2 * Sl[:, 0] * wr)
    St0 = {c: c0[np.searchsorted(r, c) - 1 if c < r[-1] else -1] for c in cuts}
    Ft = {c: np.zeros(len(Qs)) for c in cuts}
    for iq, Qv in enumerate(Qs):
        qf = Qv / HBARC
        sel = r <= Nosc / qf                         # integrate over at most Nosc oscillations
        rr = r[sel]
        integ = np.zeros(sel.sum())
        for il, l in enumerate(ls):
            if l > 0 and np.abs(Sl[sel, il]).max() < 1e-14 * np.abs(Sl[sel, 0]).max():
                continue
            integ += coef[il] * spherical_jn(l, qf * rr) * Sl[sel, il]
        cum = np.cumsum(rr**2 * integ * wr[sel])
        for c in cuts:
            Ft[c][iq] = cum[min(np.searchsorted(rr, c), len(rr)) - 1]
    out = {c: (Ft[c] / St0[c])**2 for c in cuts}
    # analytic extension of S~(0) beyond max(cuts), using the local power law of S_0 over the last decade
    # (no extension if the source has no power-law tail and has underflowed to zero)
    sel = r > max(cuts) / 10
    if np.all(Sl[sel, 0] > 0):
        sl = np.polyfit(np.log(r[sel]), np.log(Sl[sel, 0]), 1)[0]
        tail = r[-1]**3 * Sl[-1, 0] / (-3 - sl)
    else:
        sl, tail = np.nan, 0.0
    out[np.inf] = (Ft[max(cuts)] / (St0[max(cuts)] + tail))**2
    out['tailslope'] = sl
    out['S0'] = {**St0, np.inf: St0[max(cuts)] + tail}     # S~(0) for each cut, and with the extension
    return out


def levy_fit(Qs, y, window=(0.1, 0.9)):
    """alpha and R [fm] from a linear fit of ln(-ln y) vs ln Q over window[0] < y < window[1]."""
    sel = (y > window[0]) & (y < window[1])
    A = np.polyfit(np.log(Qs[sel]), np.log(-np.log(y[sel])), 1)
    return A[0], HBARC * np.exp(A[1] / A[0])
