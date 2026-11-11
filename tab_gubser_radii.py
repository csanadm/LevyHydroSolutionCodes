"""Dressed Gubser flow, isothermal dressing: Gaussian (saddle-point) homogeneity lengths.

Reproduces Table "tab:gauss" of the paper (q^-1 = 4.3 fm, tau_f = 8 fm, T_f = 140 MeV, pions), computed with the
exponential approximation K_1(A) ~ e^(-A), and the numbers quoted in its caption: the exact-K_1 values at
m_T = 0.2 GeV, and the contribution of the emission time to R_out^2, beta_T^2 Var(t) - 2 beta_T Cov(x_out, t),
with t = tau_f cosh(eta_s) on the freeze-out hypersurface (computed for the full isothermal source).
Usage: python tab_gubser_radii.py
"""
import gubser as G

if __name__ == '__main__':
    print('Table tab:gauss (asymptotic K_1); lengths in fm')
    print('m_T [GeV]   r_s   v_perp(r_s)  R_side  R_out  R_long  R_side^-2 [fm^-2]')
    for mT in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        rs, v, Rs, Ro, Rl = G.gauss_radii(mT)
        print(f'  {mT:.1f}      {rs:5.2f}   {v:.3f}     {Rs:5.2f}  {Ro:5.2f}  {Rl:5.2f}   {Rs**-2:.3f}')
    rs, v, Rs, Ro, Rl = G.gauss_radii(0.2, exact_K1=True)
    print(f'\nexact K_1 at m_T = 0.2 GeV: r_s = {rs:.2f} fm, v_perp = {v:.3f}, R_side = {Rs:.2f} fm, '
          f'R_out = {Ro:.2f} fm, R_long = {Rl:.2f} fm')
    print('\nemission-time contribution to R_out^2 [fm^2], and R_out including it [fm]')
    for mT in (0.2, 0.5, 0.8):
        dt2, var_t, cov = G.emission_time_term(mT)
        Ro = G.gauss_radii(mT)[3]
        print(f'  m_T = {mT:.1f} GeV:  {dt2:.2f}  (Var t = {var_t:.2f} fm^2, Cov(x, t) = {cov:+.3f} fm^2);  '
              f'R_out = sqrt({Ro:.2f}^2 + {dt2:.2f}) = {(Ro**2 + dt2)**0.5:.2f} fm')
