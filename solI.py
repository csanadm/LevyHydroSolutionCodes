"""Solution I (self-similar potential flow) in Milne form.

Phi = tau^lambda G(eta),  t = tau cosh(eta), r = tau sinh(eta).
Field equation d_mu (w^(k-1) d^mu Phi) = 0 with w = tau^(lambda-1) h,
h^2 = lambda^2 G^2 - G'^2, reduces to

  G'' [1 - (k-1) G'^2/h^2] = lambda p G - 2 coth(eta) G' - (k-1) lambda^2 G G'^2/h^2,
  p = 3 + k (lambda - 1).

Relative rapidity to Hubble flow: tanh(Delta) = -G'/(lambda G);  v = tanh(eta + Delta).
Proper acceleration (outward):  tau a = -[(lambda-1) sinh Delta + cosh Delta * h'/h].
F(s) = G / cosh^lambda(eta), s = tanh^2 eta  (so F_1 = k lambda (lambda-1)/6 F_0 is recovered).
"""
import numpy as np
from scipy.integrate import solve_ivp


def rhs(eta, y, lam, k):
    G, Gp = y
    p = 3 + k * (lam - 1)
    h2 = lam**2 * G**2 - Gp**2
    num = lam * p * G - 2 * Gp / np.tanh(eta) - (k - 1) * lam**2 * G * Gp**2 / h2
    den = 1 - (k - 1) * Gp**2 / h2
    return [Gp, num / den]


def solve(lam, k, eta_max=10.0, eta0=1e-3, n=4000):
    """Integrate the regular branch from the centre. Returns dict with arrays
    and a flag 'sonic' if the branch hits v_rel = c_s before eta_max."""
    G0 = np.sign(lam) if lam != 0 else 1.0
    p = 3 + k * (lam - 1)
    Gpp0 = lam * p * G0 / 3.0
    y0 = [G0 + 0.5 * Gpp0 * eta0**2, Gpp0 * eta0]
    cs = 1 / np.sqrt(k)

    def sonic(eta, y, lam, k):
        G, Gp = y
        return abs(Gp / (lam * G)) - cs * (1 - 1e-4)
    sonic.terminal = True

    ev = np.linspace(eta0, eta_max, n)
    sol = solve_ivp(rhs, (eta0, eta_max), y0, args=(lam, k), method='DOP853',
                    rtol=1e-11, atol=1e-13, events=sonic, dense_output=True)
    eta_end = sol.t[-1]
    eta = ev[ev <= eta_end]
    G, Gp = sol.sol(eta)
    Gpp = np.array([rhs(e, [g, gp], lam, k)[1] for e, g, gp in zip(eta, G, Gp)])
    h = np.sqrt(lam**2 * G**2 - Gp**2)
    hp = (lam**2 * G * Gp - Gp * Gpp) / h
    tD = -Gp / (lam * G)
    D = np.arctanh(tD)
    v = np.tanh(eta + D)
    xi = np.tanh(eta)
    taua = -((lam - 1) * np.sinh(D) + np.cosh(D) * hp / h)
    F = G / np.cosh(eta)**lam
    return dict(eta=eta, G=G, Gp=Gp, h=h, w_norm=h / h[0], vrel=np.abs(tD), v=v, xi=xi,
                r_over_tau=np.sinh(eta), taua=taua, F=F / F[0],
                sonic=len(sol.t_events[0]) > 0, eta_end=eta_end, c=k * (1 - lam) / 3)


def delta_analytic(lam, k):
    disc = 1 + k**2 * lam * (lam - (1 - 3 / k))
    if disc < 0:
        return np.nan
    return (-1 + np.sqrt(disc)) / k
