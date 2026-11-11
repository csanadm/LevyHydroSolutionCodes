"""lambda-PF: the critical (sonic) point, cases (ii) and (iii) of the paper.

1. Type of the critical point (Eq. node): node for (lambda - 1)[(3k^2 - 6k - 1) lambda - (3k^2 - 14k + 7)] < 0,
   focus otherwise. Width of the node strip below the window, between lambda_n = (3k^2-14k+7)/(3k^2-6k-1) and the
   lower window edge lambda_-, for kappa = 3, 4, 5, and the kappa where it closes (3 + 2 sqrt(2)).
2. Fate of the regular branch outside the window: does it run into the critical point (both numerator and
   denominator of the field equation vanish; 'node entry') or does it reach the sonic line with a non-vanishing
   numerator, where u' diverges ('fold')? Criterion: u' = du/deta at the sonic point agrees with the slow
   eigendirection of the node (node entry) or grows without bound (fold).
3. kappa = 3, lambda = 0.7: F''/F at s_c = 100/147 along the branch that enters the node.
4. Continuation of the transonic solutions through the node for kappa = 3, lambda = 0.6-0.9 and kappa = 5,
   lambda = 0.75-0.9: the regular branch is continued along the eigendirection of the node along which it
   arrives, and integrated to s -> 1 with the first-order equation written for the relative rapidity zeta,
   zeta' = -N/(1 - kappa tanh^2 zeta), u = -tanh(zeta). Prints v and w(t, r)/w(t, 0) for r -> t.
Run time: about 10 s.  Usage: python lambdaPF_critical.py
"""
import numpy as np
from scipy.integrate import solve_ivp
from lambdaPF import solve


def lam_minus(k):
    return (k - 3 - 2 / np.sqrt(k)) / (k - 1)


def lam_node(k):
    return (3 * k**2 - 14 * k + 7) / (3 * k**2 - 6 * k - 1)


def is_node(lam, k):
    return (lam - 1) * ((3 * k**2 - 6 * k - 1) * lam - (3 * k**2 - 14 * k + 7)) < 0


def N(eta, u, lam, k):
    return k * lam - k + 3 - 2 * u / np.tanh(eta) - k * lam * u**2


def critical_point(lam, k):
    """(eta_c, u_c) from Eq. sc; u_c = +c_s for (kappa-1) lambda > kappa - 3 (fluid slower than the pattern)."""
    X = (k - 1) * lam - (k - 3)
    sc = 4 / (k * X**2)
    return np.arctanh(np.sqrt(sc)), np.sign(X) / np.sqrt(k)


def node_slopes(lam, k):
    """Slopes du/deta of the two eigendirections of the autonomous system
    d eta/d sigma = 1 - kappa u^2,  du/d sigma = (1 - u^2) N  at the critical point (slow one first)."""
    ec, uc = critical_point(lam, k)
    J = np.array([[0.0, -2 * k * uc],
                  [(1 - uc**2) * 2 * uc / np.sinh(ec)**2, (1 - uc**2) * (-2 / np.tanh(ec) - 2 * k * lam * uc)]])
    ev, vec = np.linalg.eig(J)
    if np.iscomplexobj(ev) and np.any(np.abs(ev.imag) > 0):
        return None
    order = np.argsort(np.abs(ev.real))
    return [vec[1, i].real / vec[0, i].real for i in order]


def classify(lam, k):
    """Integrate the regular branch until it becomes sonic (v_rel = c_s (1 - 1e-4)).
    Returns eta_end - eta_c, N and u' = du/deta at the end, and the solution dict."""
    s = solve(lam, k, eta_max=10.0, n=40000)
    if not s['sonic']:
        return None
    ec, uc = critical_point(lam, k)
    e = s['eta']
    u = s['Gp'] / (lam * s['G'])
    n_end = N(e[-1], u[-1], lam, k)
    up_end = (1 - u[-1]**2) * n_end / (1 - k * u[-1]**2)
    return e[-1] - ec, n_end, up_end, s


if __name__ == '__main__':
    print('1. node strip below the window')
    for k in (3, 4, 5):
        print(f'kappa={k}: node for lambda_n = {lam_node(k):+.4f} < lambda < lambda_- = {lam_minus(k):+.4f}; '
              f'width {lam_minus(k) - lam_node(k):.4f}')
    ks = 3 + 2 * np.sqrt(2)
    print(f'kappa = 3 + 2 sqrt 2 = {ks:.4f}: lambda_- - lambda_n = {lam_minus(ks) - lam_node(ks):.1e} (the strip closes)')
    for k in (6, 7, 9):
        lH = 1 - 3 / k
        lc = 0.5 * (lH + np.sqrt(lH**2 - 4 / k**2))
        print(f'kappa={k}: lower window edge lambda_c = {lc:.4f} < lambda_n = {lam_node(k):.4f}: no node strip below the window')

    print('\n2. fate of the regular branch outside the window (eta_end - eta_c, N at the sonic point)')
    scans = {3: [-0.99, -0.9, -0.8, -0.7, -0.6, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.05, 1.2],
             5: [0.2728, 0.2732, 0.2736, 0.2740, 0.2744, 0.2748, 0.2752, 0.2760, 0.75, 0.8, 0.9, 0.99, 1.05, 1.3]}
    for k, lams in scans.items():
        for lam in lams:
            res = classify(lam, k)
            typ = 'node' if is_node(lam, k) else 'focus'
            if res is None:
                print(f'kappa={k} lambda={lam:+.4f}: reaches the light cone')
                continue
            d, n_end, up_end, _ = res
            slopes = node_slopes(lam, k)
            slow = slopes[0] if slopes is not None else np.nan
            entry = slopes is not None and abs(up_end - slow) < 0.05 * abs(slow) + 1e-3
            print(f'kappa={k} lambda={lam:+.4f} ({typ:5s}): eta_end - eta_c = {d:+.1e}, N = {n_end:+.1e}, '
                  f"u' = {up_end:+.3f}, slow eigen-slope = {slow:+.3f}  -> {'node entry' if entry else 'fold'}")

    print('\n3. kappa = 3, lambda = 0.7: F\'\'/F at s_c')
    k, lam = 3, 0.7
    ec, uc = critical_point(lam, k)
    d, n_end, slope_num, _ = classify(lam, k)
    slopes = node_slopes(lam, k)
    up = min(slopes, key=lambda x: abs(x - slope_num))       # the branch arrives along the slow eigendirection
    th, ch = np.tanh(ec), np.cosh(ec)
    a = lam * uc - lam * th                                # d ln F / d eta
    ap = lam * up - lam / ch**2                            # its derivative
    s1 = 2 * th / ch**2                                    # ds/d eta
    s2 = 2 / ch**4 - 4 * th**2 / ch**2                     # d^2 s / d eta^2
    Fpp = (ap + a**2) / s1**2 - a * s2 / s1**3
    print(f's_c = {np.tanh(ec)**2:.6f} (100/147 = {100 / 147:.6f}); eigen-slopes du/deta = '
          + ', '.join(f'{x:.4f}' for x in slopes) + f"; u' of the branch at the sonic point {slope_num:.4f}; F''/F = {Fpp:.3f}")

    print('\n4. continuation through the node to s -> 1')
    for k, lams in ((3, (0.6, 0.7, 0.8, 0.9)), (5, (0.75, 0.8, 0.85, 0.9))):
        for lam in lams:
            d, n_end, slope_num, s = classify(lam, k)
            slopes = node_slopes(lam, k)
            up = min(slopes, key=lambda x: abs(x - slope_num))
            ec, uc = critical_point(lam, k)
            eps = 1e-4
            u0 = uc + up * eps
            # ln G at the critical point from the regular branch, extrapolated with (ln G)' = lambda u
            lnG = np.log(abs(s['G'][-1])) + lam * uc * (ec - s['eta'][-1])
            f = lambda e, y: [-N(e, -np.tanh(y[0]), lam, k) / (1 - k * np.tanh(y[0])**2), -lam * np.tanh(y[0])]
            sol = solve_ivp(f, (ec + eps, 25.0), [-np.arctanh(u0), lnG + lam * u0 * eps],
                            method='DOP853', rtol=1e-11, atol=1e-13, dense_output=True)
            out = []
            for e in (8.0, 15.0, 25.0):
                zeta, lg = sol.sol(e)
                # w at fixed t = 1: tau = 1/cosh(eta), w = tau^(lambda-1) h, h = |lambda| G sqrt(1 - u^2) = |lambda| G / cosh(zeta)
                lnw = (lam - 1) * np.log(1 / np.cosh(e)) + np.log(abs(lam)) + lg - np.log(np.cosh(zeta))
                lnw0 = np.log(abs(lam))                    # centre: tau = t = 1, G = 1, u = 0
                out.append((np.tanh(e + zeta), np.exp(lnw - lnw0)))
            print(f'kappa={k} lambda={lam:.2f}: v, w(t,r)/w(t,0) at eta = 8 / 15 / 25: '
                  + ' / '.join(f'({v:.4f}, {w:.4g})' for v, w in out))
