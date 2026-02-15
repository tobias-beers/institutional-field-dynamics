"""
Institutional Field Dynamics v4: Relational, Dynamic, Attractor-Aware

Core question: "Does the configuration produce cooperation or conflict?"

This model analyzes whether a policy configuration creates cooperation (positive α)
or conflict (negative α), and how the power structure W evolves as a consequence.

Extensions over v3:
1. Relational U: utility depends on proximity to other actors (α)
2. Dynamic W: power matrix evolves based on cooperation/conflict
3. Attractor analysis: explicit test of multiple initializations

Usage:
    from beleidsdynamica_v4 import Actor, Krachtenveld, analyse

    actors = [
        Actor("Ministry", np.array([3.5, 6.0, 5.0])),
        Actor("Teacher",  np.array([4.0, 6.5, 5.5])),
    ]

    U_config = {
        "Ministry": {
            'doel': np.array([6.0, 4.0, 4.0]),
            'gewicht': 0.5,
            'alpha': {'Teacher': 0.0}
        },
        ...
    }

    system = Krachtenveld(actors, U_config, W, eta=0.01)
    result = analyse.diagnose(system)
"""

from .model import Actor, Krachtenveld
from . import analyse
from . import visualisatie

__version__ = "0.4.0"
__all__ = ["Actor", "Krachtenveld", "analyse", "visualisatie"]
