# LevyHydroSolutionCodes

Python codes accompanying the paper

> M. Csanád, M. Nagy, B. Szentkuti, Á. Benyó,
> *Accelerating exact solutions of relativistic hydrodynamics and Lévy-shaped freeze-out sources*,
> arXiv:XXXX.XXXXX

The repository contains the numerical integration of the self-similar **λ potential flows (λ-PF)**, the
source function and Lévy fits of the **isothermally dressed Gubser flow**, and the scripts that produce
the figures of the paper.

## Requirements

Python 3 with NumPy, SciPy and Matplotlib (tested with Python 3.13, NumPy 2.x, SciPy 1.18, Matplotlib 3.11).

```bash
pip install numpy scipy matplotlib
```

## Contents

In the file names, `solI` refers to λ-PF and `solII`/`gubser` to the dressed Gubser flow
(earlier working names of the two solutions).

| File | Contents |
|------|----------|
| `solI.py` | λ-PF in Milne form, Φ = τ<sup>λ</sup> G(η): right-hand side of the field equation, integration of the regular branch from the centre (with detection of the sonic line), analytic light-cone exponent δ |
| `gubser.py` | Gubser flow (velocity, pressure, proper acceleration), transverse source with isothermal or isentropic dressing, side-direction Lévy fit `levy_fit_side` |
| `gubser_obs.py` | Benchmark parameters (q<sup>−1</sup> = 4.3 fm, τ<sub>f</sub> = 8 fm, T<sub>f</sub> = 140 MeV) and the isothermally dressed temperature T(τ, r) |
| `make_levy_data.py` | Computes the Lévy fits of the dressed Gubser flow (α and R vs. m<sub>T</sub>) and stores them in `figs/*.npy` |
| `fig_paper_solI.py` | λ-PF figures: profiles, (κ, λ) window, proper acceleration |
| `fig_paper_gubser_Tearly.py` | Dressed Gubser flow: temperature at early times |
| `fig_paper_solII_levy.py` | Dressed Gubser flow: Lévy exponent α and R<sup>−2</sup> vs. m<sub>T</sub> |

Figures are written to `figures/` (PDF as used in the paper, plus a PNG preview).
Intermediate data are written to `figs/`.

## Reproducing the figures

```bash
mkdir -p figs

# lambda-PF: profiles, window scan, acceleration
python fig_paper_solI.py                    # or select panels: profiles window accel

# dressed Gubser flow: early-time temperature
python fig_paper_gubser_Tearly.py

# dressed Gubser flow: Levy fits (data first, then the plot)
python make_levy_data.py
python fig_paper_solII_levy.py
```

| Output | Paper figure | Run time* |
|--------|--------------|-----------|
| `figures/fig_solI_profiles.pdf` | regular branch of λ-PF for κ = 3 and 5 | seconds |
| `figures/fig_solI_window.pdf` | fate of the regular branch in the (κ, λ) plane | ~3 min |
| `figures/fig_solI_accel.pdf` | proper acceleration of λ-PF | seconds |
| `figures/fig_gubser_Tearly.pdf` | dressed Gubser flow at early times | seconds |
| `figures/fig_solII_levy.pdf` | Lévy fits of the dressed Gubser flow | ~1.5 min (data) |

\*single core, recent laptop/server

## Examples

**λ potential flows.** Regular branch for κ = 3 (top) and κ = 5 (bottom): potential, radial velocity,
velocity relative to the similarity lines, and w = p<sup>1/(κ+1)</sup> on τ = const. Crosses mark where the
branch reaches the sonic line.

<img width="700" alt="lambpda-PF profiles" src="https://github.com/user-attachments/assets/60a655c2-26fd-4dc2-8dd7-d507b59c41c5" />

**Dressed Gubser flow.** Lévy exponent α (top) and R<sup>−2</sup> (bottom) versus m<sub>T</sub> for pions,
from side-direction fits of the transverse source.

<img width="420" alt="Levy fits of the dressed Gubser flow" src="https://github.com/user-attachments/assets/9b5bb9c0-ff84-4b48-8cb2-b38a1c0b16ad" />

The images above illustrate the output of the code; the paper is the reference for all results.

## Using the solvers directly

```python
from solI import solve
sol = solve(lam=0.3, k=5)        # regular branch of lambda-PF for kappa = 5
print(sol.keys())

import gubser as G
alpha, R = G.levy_fit_side(mT=0.4, q=1/4.3, tauf=8.0, Tf=0.140, dressing='iso')[:2]
print(alpha, R)                  # side-direction Levy exponent and scale (fm)
```

## Citation

If you use these codes, please cite the paper above. The version of the code used for the paper is tagged
as `v1.0`.

## License

[to be added]
