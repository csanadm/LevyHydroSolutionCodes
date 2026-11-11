"""Excerpt of claude/plotcode/gubser_obs.py needed by fig_paper_gubser_Tearly.py (parameters and T_iso, unchanged)."""
import numpy as np
import gubser as G

Q, TAUF, TF = 1 / 4.3, 8.0, 0.140


def T_iso(tau, r):
    """Eq. (Tiso): T = T_f (tau_f/tau) [cosh rho_f(theta) / cosh rho(tau, r)]^(2/3)."""
    cot = (1 + Q**2 * tau**2 - Q**2 * r**2) / (2 * Q * r)
    qrf = -cot + np.sqrt(cot**2 + 1 + Q**2 * TAUF**2)
    rf = qrf / Q
    chf = np.sqrt(G.D(TAUF, rf, Q)) / (2 * Q * TAUF)
    ch = np.sqrt(G.D(tau, r, Q)) / (2 * Q * tau)
    return TF * (TAUF / tau) * (chf / ch)**(2 / 3), rf
