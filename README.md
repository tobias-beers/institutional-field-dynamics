# Institutional Field Dynamics

A force field model of multi-actor governance. Formalizes institutional dynamics as a system in which actors move through configuration space under three types of force: intrinsic motivation, social influence, and institutional pressure.

**Paper:** Beers, T.J. (2026). *Institutional Field Dynamics: A Force Field Model of Multi-Actor Governance.* SSRN Working Paper.

## What this does

The model treats policy not as direct instruction but as transformation of the force field:

$$P: (U, W, C, \Phi) \rightarrow (U', W', C', \Phi')$$

It generates computable diagnostics — coherence, friction, power asymmetry, attractor count — that characterize the structural health of institutional configurations.

## Installation

```bash
pip install -e .
```

Requires Python 3.9+ with numpy and matplotlib.

## Core classes

- `Krachtenveld` — the force field system (actors, preferences, power matrix, constraints)
- `Actor` — an agent in institutional configuration space
- `diagnose()` — computes structural diagnostics (coherence, asymmetry, attractors)

## Quick example

```python
from beleidsdynamica_v4 import Krachtenveld, Actor
from beleidsdynamica_v4.analyse import diagnose
import numpy as np

# Two actors in 2D configuration space
actors = [
    Actor("Ministry", np.array([3.0, 7.0]), '#e74c3c'),
    Actor("Teacher",  np.array([6.0, 4.0]), '#27ae60'),
]

config = {
    "Ministry": {'doel': np.array([5.0, 8.0]), 'gewicht': 0.5, 'alpha': {'Teacher': -0.03}},
    "Teacher":  {'doel': np.array([3.0, 7.0]), 'gewicht': 0.4, 'alpha': {'Ministry': -0.05}},
}

W = np.array([[0.0, 0.1],
              [0.4, 0.0]])  # Ministry barely hears Teacher

system = Krachtenveld(actors, config, W)
system.simuleer(200)

d = np.array([-1, 1]) / np.sqrt(2)
result = diagnose(system, d_gewenst=d)
print(f"Asymmetry: {result.asymmetrie_W:.3f}")
print(f"Attractors: {result.n_attractoren}")
```

## Experiments

The `experiments/` directory contains the computational experiments from the paper:

| Script | Description |
|--------|-------------|
| `ssrn_sensitivity.py` | W-perturbation, alpha-sweep, multi-expert robustness (SSRN paper Section 4.6) |

Run any experiment:

```bash
cd experiments
python ssrn_sensitivity.py
```

Figures are saved to `experiments/figures/`.

## Citation

```bibtex
@article{beers2026institutional,
  title={Institutional Field Dynamics: A Force Field Model of Multi-Actor Governance},
  author={Beers, Tobias J.},
  year={2026},
  journal={SSRN Working Paper}
}
```

## License

MIT
