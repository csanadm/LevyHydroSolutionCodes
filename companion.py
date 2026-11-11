"""Companion solutions with 1/r acceleration (Appendix "Companion solutions" of the paper).

Self-similar flows with power-law initial data at rest, p_0 ∝ r^(-a_p): with xi = r/t and v = V(xi),
    V' = [2(kappa+1)(1 - V xi) V/(kappa xi) - a_p (1 - V^2)] / [gamma^2 (kappa+1) ((V - xi)^2 - (1 - V xi)^2/kappa)].
1. Regular centre branch V ≃ c xi, c = kappa a_p/[3(kappa+1)]: it coincides with the regular branch of lambda-PF with
   lambda = 1 - a_p/(kappa+1) (comparison with lambdaPF.solve); for kappa = 3 it reaches the light cone
   subsonically for 1.69 < a_p < 6.31 (scan); a_p = 3(kappa+1)/kappa gives exactly V = xi.
2. Tail branch, V ≃ a_p/[(kappa+1) xi] for xi >> 1, integrated inwards: V = 1/xi exactly for a_p = kappa + 1;
   for a_p > kappa + 1 it reaches V -> 1 as xi -> 1+, for a_p < kappa + 1 it crosses the light cone and becomes
   sonic just inside it.
3. Exact spherical Rindler atmosphere, u = (r, t)/rho, rho = sqrt(r^2 - t^2) (symbolic, needs sympy): proper
   acceleration 1/rho; with nu ∝ 1/rho the temperature is uniform at t = 0 and n ∝ r^-(kappa+1), i.e. alpha = kappa - 2.
Run time: a few seconds.  Usage: python companion.py
"""
import numpy as np
from scipy.integrate import solve_ivp
from lambdaPF import solve


def Vprime(xi, V, k, ap):
    num = 2 * (k + 1) * (1 - V * xi) * V / (k * xi) - ap * (1 - V**2)
    den = (k + 1) / (1 - V**2) * ((V - xi)**2 - (1 - V * xi)**2 / k)
    return num / den


def centre_branch(k, ap, xi_max=1 - 1e-6):
    """Integrate the regular centre branch outwards; stops if it becomes sonic. Returns the solution and a flag."""
    c = k * ap / (3 * (k + 1))
    x0 = 1e-4

    def sonic(xi, y, *a):
        V = y[0]
        return (V - xi)**2 - (1 - V * xi)**2 / k
    sonic.terminal = True
    sol = solve_ivp(lambda x, y: [Vprime(x, y[0], k, ap)], (x0, xi_max), [c * x0], method='DOP853',
                    rtol=1e-11, atol=1e-13, events=sonic, dense_output=True)
    return sol, len(sol.t_events[0]) == 0


def tail_branch(k, ap, x0=1e4):
    """Integrate the tail branch inwards from xi = x0 until it becomes sonic or V reaches 1."""
    def sonic(xi, y, *a):
        V = y[0]
        return (V - xi)**2 - (1 - V * xi)**2 / k
    sonic.terminal = True

    def vone(xi, y, *a):
        return 1 - 1e-6 - y[0]
    vone.terminal = True
    return solve_ivp(lambda x, y: [Vprime(x, y[0], k, ap)], (x0, 0.5), [ap / ((k + 1) * x0)], method='DOP853',
                     rtol=1e-11, atol=1e-14, events=(sonic, vone))


if __name__ == '__main__':
    print('1. centre branch vs lambda-PF with lambda = 1 - a_p/(kappa+1)')
    for k, ap in ((3, 3.0), (3, 4.5), (5, 4.0), (5, 3.5)):
        sol, ok = centre_branch(k, ap)
        s = solve(1 - ap / (k + 1), k)
        sel = (s['xi'] > 1e-3) & (s['xi'] < 0.99)
        dv = np.max(np.abs(sol.sol(s['xi'][sel])[0] - s['v'][sel]))
        print(f'kappa={k} a_p={ap}: reaches the light cone: {ok};  max |V - v_lambdaPF| for xi < 0.99: {dv:.1e}')
    k = 3
    ok = [ap for ap in np.arange(1.0, 7.01, 0.01) if centre_branch(k, ap)[1]]
    print(f'kappa=3: the centre branch reaches the light cone for {min(ok):.2f} <= a_p <= {max(ok):.2f}  '
          f'(analytic: {4 * (1 - 1 / np.sqrt(3)):.2f} < a_p < {4 * (1 + 1 / np.sqrt(3)):.2f})')
    for k in (3, 5):
        ap = 3 * (k + 1) / k
        sol, _ = centre_branch(k, ap)
        xs = np.linspace(0.01, 0.99, 50)
        print(f'kappa={k} a_p = 3(kappa+1)/kappa = {ap:.3f}: max |V - xi| = {np.max(np.abs(sol.sol(xs)[0] - xs)):.1e}')

    print('2. tail branch, integrated inwards from xi = 10^4')
    for k in (3, 5):
        for ap in (k + 1.0, k + 1.5, k + 0.5, k - 0.5):
            sol = tail_branch(k, ap)
            xi_e, V_e = sol.t[-1], sol.y[0, -1]
            if sol.t_events[0].size:
                end = f'sonic at xi = {xi_e:.5f}, V = {V_e:.4f}'
            elif sol.t_events[1].size:
                end = f'V -> 1 at xi = {xi_e:.6f}'
            elif sol.status == -1:
                D = (V_e - xi_e)**2 - (1 - V_e * xi_e)**2 / k
                end = (f"crosses the light cone and folds (V' diverges) at xi = {xi_e:.5f}, V = {V_e:.4f}; "
                       f'distance from the sonic line {D:.1e}')
            else:
                end = f'reached xi = {xi_e:.3f}, V = {V_e:.4f}'
            extra = ''
            if ap == k + 1:
                xs = np.geomspace(1.05, 1e3, 40)
                sol2 = solve_ivp(lambda x, y: [Vprime(x, y[0], k, ap)], (1e4, 1.05), [1e-4], t_eval=xs[::-1],
                                 method='DOP853', rtol=1e-12, atol=1e-15)
                extra = f';  max |V xi - 1| for xi = 1.05-10^3: {np.max(np.abs(sol2.y[0] * sol2.t - 1)):.1e}'
            print(f'kappa={k} a_p={ap:.1f} (kappa+1 = {k + 1}): {end}{extra}')

    print('3. Rindler atmosphere (symbolic)')
    try:
        import sympy as sp
    except ImportError:
        sp = None
        print('sympy not installed: skipped')
    if sp is not None:
        t, r, kap = sp.symbols('t r kappa', positive=True)
        rho = sp.sqrt(r**2 - t**2)
        ut, ur = r / rho, t / rho
        at = ut * sp.diff(ut, t) + ur * sp.diff(ut, r)
        ar = ut * sp.diff(ur, t) + ur * sp.diff(ur, r)
        print('u.u =', sp.simplify(ut**2 - ur**2), ';  -a.a rho^2 =', sp.simplify((ar**2 - at**2) * rho**2),
              ' (proper acceleration 1/rho)')
        T = (rho / r)**(2 / kap)                         # T = T_0 (rho/r)^(2/kappa)
        p = (1 / rho)**(kap + 1) * (rho / r)**(2 * (kap + 1) / kap)
        print('at t = 0: T/T_0 =', sp.simplify(T.subs(t, 0)), ';  n ∝ p/T =', sp.simplify((p / T).subs(t, 0)),
              ' -> r^-(3+alpha) with alpha = kappa - 2')
