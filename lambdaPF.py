"""lambda potential flows (lambda-PF): solver for the regular branch.

The self-similar potential Phi = t^lambda F(s), s = r^2/t^2, is written in Milne form,

    Phi = tau^lambda G(eta),   t = tau cosh(eta),  r = tau sinh(eta),  G = F cosh^lambda(eta).

With w = p^(1/(kappa+1)) = tau^(lambda-1) h and h^2 = lambda^2 G^2 - G'^2, the field equation
d_mu (w^(kappa-1) d^mu Phi) = 0 reduces to

    G'' [1 - (kappa-1) G'^2/h^2] = lambda P G - 2 coth(eta) G' - (kappa-1) lambda^2 G G'^2/h^2,
    P = 3 + kappa (lambda - 1) = kappa (lambda - lambda_H),   lambda_H = 1 - 3/kappa.

Kinematics, with zeta the rapidity of the flow relative to Hubble flow:
    tanh(zeta) = -G'/(lambda G),   v_rel = |tanh(zeta)|,   v = tanh(eta + zeta),
    outward proper acceleration on tau = const:  tau a = -[(lambda-1) sinh(zeta) + cosh(zeta) h'/h].
The normalisation is |F(0)| = 1 with the sign of lambda; the central expansion reproduces
F_1 = kappa lambda (lambda-1) F_0 / 6.
"""
import numpy as np
from scipy.integrate import solve_ivp


def rhs(eta, y, lam, k):
    """Right-hand side of the field equation for y = (G, G'); k = kappa."""
    G, Gp = y
    P = 3 + k * (lam - 1)
    h2 = lam**2 * G**2 - Gp**2
    num = lam * P * G - 2 * Gp / np.tanh(eta) - (k - 1) * lam**2 * G * Gp**2 / h2
    den = 1 - (k - 1) * Gp**2 / h2
    return [Gp, num / den]


def solve(lam, k, eta_max=10.0, eta0=1e-3, n=4000):
    """Integrate the regular branch from the centre up to eta_max (r/tau = sinh(eta_max)).

    The integration stops if the branch reaches the sonic line v_rel = c_s = 1/sqrt(kappa).
    Returns a dict with arrays on an eta grid of n points:
      eta, G, Gp, h            Milne-form potential, its derivative, and h = w tau^(1-lambda)
      w_norm                   w(tau, r) / w(tau, 0) on tau = const
      F                        F(s) / F(0) with s = tanh^2(eta)
      vrel, v, xi, r_over_tau  relative velocity, radial velocity, r/t, r/tau
      taua                     tau a, outward proper acceleration on tau = const
    and the scalars sonic (True if the sonic line was reached), eta_end, c = kappa (1-lambda)/3.
    """
    G0 = np.sign(lam) if lam != 0 else 1.0
    P = 3 + k * (lam - 1)
    Gpp0 = lam * P * G0 / 3.0            # regularity at the centre
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
    tz = -Gp / (lam * G)                 # tanh(zeta)
    zeta = np.arctanh(tz)
    v = np.tanh(eta + zeta)
    xi = np.tanh(eta)
    taua = -((lam - 1) * np.sinh(zeta) + np.cosh(zeta) * hp / h)
    F = G / np.cosh(eta)**lam
    return dict(eta=eta, G=G, Gp=Gp, h=h, w_norm=h / h[0], vrel=np.abs(tz), v=v, xi=xi,
                r_over_tau=np.sinh(eta), taua=taua, F=F / F[0],
                sonic=len(sol.t_events[0]) > 0, eta_end=eta_end, c=k * (1 - lam) / 3)


def delta_analytic(lam, k):
    """Light-cone exponent delta of the regular branch, Phi ~ tau^lambda e^(delta eta) for large eta:
    kappa lambda (lambda - lambda_H) = delta (2 + kappa delta), root delta_+. Returns nan if delta is complex."""
    disc = 1 + k**2 * lam * (lam - (1 - 3 / k))
    if disc < 0:
        return np.nan
    return (-1 + np.sqrt(disc)) / k


if __name__ == '__main__':
    # checks against numbers quoted in the paper: light-cone exponent delta (numerical vs. analytic),
    # sonic termination, and the asymptotic acceleration tau a
    for k, lam in [(5, 0.3), (5, 0.35), (5, 0.5), (5, 0.65), (3, -0.5), (3, 0.3), (3, 0.5), (3, 0.7), (5, 0.2), (5, 0.8)]:
        s = solve(lam, k)
        dn = np.gradient(np.log(s['h']), s['eta'])[-1] if not s['sonic'] else np.nan
        print(f"kappa={k} lambda={lam:+.2f} c={s['c']:.3f} sonic={s['sonic']} eta_end={s['eta_end']:.3f} "
              f"delta_num={dn:+.5f} delta_an={delta_analytic(lam, k):+.5f} tau_a(end)={s['taua'][-1]:+.4f}")
    # location of the critical point for kappa = 3, lambda = 0.7: s_c = 100/147
    k, lam = 3, 0.7
    print(f"s_c(kappa=3, lambda=0.7) = {4 / (k * ((k - 1) * lam - (k - 3))**2):.6f}  (100/147 = {100 / 147:.6f})")
