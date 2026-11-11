"""Run every script of the repository in order and store the printed output of each in results/<script>.txt.
Total run time: about 30 minutes on a single core.  Usage: python run_all.py
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    'checks.py',                 # symbolic and numerical checks of analytic results
    'hubble_levy.py',            # Hubble-type Levy solution: source tail and high-m_T spectrum
    'lambdaPF_window.py',        # lambda-PF: subsonic window, bounds on the core acceleration
    'lambdaPF_critical.py',      # lambda-PF: critical point, node/fold, continuation through the node
    'companion.py',              # companion solutions with 1/r acceleration (appendix)
    'fig_lambdaPF.py',           # lambda-PF figures (profiles, window, acceleration)
    'fig_gubser_Tearly.py',      # dressed Gubser flow: early-time temperature
    'gubser_levy_data.py',       # dressed Gubser flow: Levy fits (data for the next script)
    'fig_gubser_levy.py',        # dressed Gubser flow: Levy-fit figure
    'tab_gubser_spectra.py',     # dressed Gubser flow: spectra, q scan
    'tab_gubser_radii.py',       # dressed Gubser flow: Gaussian radii
    'gubser_kaons.py',           # dressed Gubser flow: kaons vs pions
    'gubser_fit_systematics.py', # dressed Gubser flow: Levy-fit systematics
    'tab_lambdaPF_tails.py',     # lambda-PF: asymptotic source tails
    'tab_lambdaPF_levy.py',      # lambda-PF: Levy fits
    'tab_lambdaPF_cuts.py',      # lambda-PF: particle number, cut dependence of the Levy fits
    'gubser_tail.py',            # dressed Gubser flow: origin of the Levy shape, out and long directions
]

if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    for script in SCRIPTS:
        t0 = time.time()
        print(f'>>> {script}', flush=True)
        env = dict(os.environ, PYTHONIOENCODING='utf-8')      # e.g. sympy output on Windows consoles
        out = subprocess.run([sys.executable, script], cwd=HERE, capture_output=True, text=True,
                             encoding='utf-8', env=env)
        with open(os.path.join(HERE, 'results', script.replace('.py', '.txt')), 'w', encoding='utf-8') as f:
            f.write(out.stdout)
        if out.returncode != 0:
            sys.exit(f'{script} failed:\n{out.stderr}')
        print(f'    done in {time.time() - t0:.0f} s, output in results/{script.replace(".py", ".txt")}', flush=True)
