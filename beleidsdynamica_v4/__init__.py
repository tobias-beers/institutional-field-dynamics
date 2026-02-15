"""
Beleidsdynamica v4: Relationeel, Dynamisch, Attractor-Bewust

Kernvraag: "Ontstaat samenwerking of conflict?"

Dit model analyseert of beleid samenwerking (positieve α) of conflict (negatieve α)
creëert, en hoe de invloedsstructuur W evolueert als gevolg.

Uitbreidingen t.o.v. v3:
1. Relationele U: nutsfunctie hangt ook af van nabijheid tot anderen (α)
2. Dynamische W: invloedsmatrix evolueert op basis van samenwerking/conflict
3. Attractor-analyse: expliciet testen van meerdere initialisaties

Gebruik:
    from beleidsdynamica_v4 import Actor, Krachtenveld, analyse

    # Definieer actoren
    actoren = [
        Actor("Uitvoerder", np.array([5.0, 5.0, 5.0])),
        Actor("Manager", np.array([5.0, 5.0, 5.0])),
    ]

    # Definieer structuur met α (relationele component)
    U_config = {
        "Uitvoerder": {
            'doel': np.array([3.0, 7.0, 7.0]),
            'gewicht': 0.5,
            'alpha': {'Manager': 0.1}  # Positief = samenwerking
        },
        ...
    }

    # Analyseer met dynamische W
    systeem = Krachtenveld(actoren, U_config, W, eta=0.01)
    diagnose = analyse.diagnose(systeem)
"""

from .model import Actor, Krachtenveld
from . import analyse
from . import visualisatie

__version__ = "0.4.0"
__all__ = ["Actor", "Krachtenveld", "analyse", "visualisatie"]
