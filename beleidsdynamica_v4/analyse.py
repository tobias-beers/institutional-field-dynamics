"""
Beleidsdynamica v4 Analyse Tools

Kernvraag: "Ontstaat samenwerking of conflict?"
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Diagnose:
    """Diagnose van een beleidssysteem (v4)."""
    asymmetrie_W: float
    coherentie: float
    effectieve_coherentie: float
    energie: float
    constructieve_spanning: float
    posities: Dict[str, np.ndarray]
    krachten: Dict[str, np.ndarray]
    alpha_matrix: np.ndarray
    n_attractoren: int
    waarschuwingen: List[str]
    samenwerking_score: float  # Nieuw in v4


def diagnose(systeem, d_gewenst: Optional[np.ndarray] = None,
             test_attractoren: bool = True) -> Diagnose:
    """
    Voer een diagnose uit op het systeem (v4).

    Parameters
    ----------
    systeem : Krachtenveld
        Het te analyseren systeem
    d_gewenst : np.ndarray, optional
        Gewenste richting. Default: [-1, 1, 1] (lager werkdruk, hoger autonomie, betere middelen)
    test_attractoren : bool
        Of attractor-analyse moet worden uitgevoerd (kan traag zijn)

    Returns
    -------
    Diagnose
        Diagnose-object met alle metrieken
    """
    if d_gewenst is None:
        d_gewenst = np.array([-1, 1, 1])
        d_gewenst = d_gewenst / np.linalg.norm(d_gewenst)

    asym = systeem.asymmetrie_W()
    coh = systeem.coherentie()
    eff_coh = systeem.effectieve_coherentie(d_gewenst)
    energie = systeem.energie()
    constr = systeem.constructieve_spanning(d_gewenst)

    posities = {a.naam: a.positie.copy() for a in systeem.actoren}
    krachten_dict = {}
    for i, a in enumerate(systeem.actoren):
        krachten_dict[a.naam] = systeem.kracht(i)

    # Attractor-analyse (nieuw in v4)
    n_attractoren = 0
    if test_attractoren:
        attractoren = systeem.vind_attractoren_multi(n_starts=10)
        n_attractoren = len(attractoren)

    # Samenwerking score: gemiddelde α (nieuw in v4)
    alpha_flat = systeem.alpha_matrix.flatten()
    alpha_flat = alpha_flat[alpha_flat != 0]  # Exclude zeros (self)
    samenwerking_score = np.mean(alpha_flat) if len(alpha_flat) > 0 else 0.0

    # Waarschuwingen genereren
    waarschuwingen = []

    if asym > 0.8:
        waarschuwingen.append(
            f"HOGE ASYMMETRIE W ({asym:.2f}): Invloedsstructuur is sterk eenzijdig."
        )

    if samenwerking_score < -0.03:
        waarschuwingen.append(
            f"CONFLICT DETECTIE: Gemiddelde α is negatief ({samenwerking_score:.3f}). "
            "Actoren willen van elkaar weg → conflict-dynamiek."
        )
    elif samenwerking_score > 0.05:
        # Dit is positief, maar we noteren het
        pass

    if n_attractoren > 3:
        waarschuwingen.append(
            f"MEERDERE ATTRACTOREN ({n_attractoren}): Systeem kan naar verschillende "
            "evenwichten gaan → onvoorspelbaar."
        )

    if coh > 0.7 and eff_coh < 0.3:
        waarschuwingen.append(
            f"COHERENTIE VERKEERDE RICHTING: Krachten coherent ({coh:.2f}) maar "
            f"niet de goede kant op (eff. coh. = {eff_coh:.2f})."
        )

    return Diagnose(
        asymmetrie_W=asym,
        coherentie=coh,
        effectieve_coherentie=eff_coh,
        energie=energie,
        constructieve_spanning=constr,
        posities=posities,
        krachten=krachten_dict,
        alpha_matrix=systeem.alpha_matrix.copy(),
        n_attractoren=n_attractoren,
        samenwerking_score=samenwerking_score,
        waarschuwingen=waarschuwingen
    )


def vergelijk_beleid(systeem_voor, systeem_na, d_gewenst: Optional[np.ndarray] = None) -> Dict:
    """
    Vergelijk systeem voor en na beleidsinterventie (v4).
    """
    diag_voor = diagnose(systeem_voor, d_gewenst)
    diag_na = diagnose(systeem_na, d_gewenst)

    delta = {
        'asymmetrie_W': diag_na.asymmetrie_W - diag_voor.asymmetrie_W,
        'coherentie': diag_na.coherentie - diag_voor.coherentie,
        'effectieve_coherentie': diag_na.effectieve_coherentie - diag_voor.effectieve_coherentie,
        'samenwerking_score': diag_na.samenwerking_score - diag_voor.samenwerking_score,
        'n_attractoren': diag_na.n_attractoren - diag_voor.n_attractoren,
    }

    interpretatie = []

    if delta['samenwerking_score'] > 0.02:
        interpretatie.append("SAMENWERKING TOEGENOMEN: α is positiever geworden")
    elif delta['samenwerking_score'] < -0.02:
        interpretatie.append("CONFLICT TOEGENOMEN: α is negatiever geworden")

    if delta['n_attractoren'] < 0:
        interpretatie.append(f"STABILITEIT VERBETERD: Minder attractoren ({diag_voor.n_attractoren} → {diag_na.n_attractoren})")
    elif delta['n_attractoren'] > 0:
        interpretatie.append(f"STABILITEIT VERSLECHTERD: Meer attractoren ({diag_voor.n_attractoren} → {diag_na.n_attractoren})")

    if delta['asymmetrie_W'] > 0.1:
        interpretatie.append("W is ASYMMETRISCHER geworden")
    elif delta['asymmetrie_W'] < -0.1:
        interpretatie.append("W is SYMMETRISCHER geworden (meer wederkerigheid)")

    return {
        'voor': diag_voor,
        'na': diag_na,
        'delta': delta,
        'interpretatie': interpretatie
    }


def print_diagnose(diag: Diagnose, titel: str = "DIAGNOSE v4"):
    """Print een diagnose in leesbaar formaat."""
    print("=" * 60)
    print(titel)
    print("=" * 60)
    print(f"\nAsymmetrie W:          {diag.asymmetrie_W:.3f}")
    print(f"Coherentie:            {diag.coherentie:.3f}")
    print(f"Effectieve coherentie: {diag.effectieve_coherentie:.3f}")
    print(f"Energie:               {diag.energie:.3f}")
    print(f"Samenwerking score (α):{diag.samenwerking_score:.3f}")
    print(f"Aantal attractoren:    {diag.n_attractoren}")

    print("\nPOSITIES:")
    for naam, pos in diag.posities.items():
        print(f"  {naam}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    if diag.waarschuwingen:
        print("\nWAARSCHUWINGEN:")
        for w in diag.waarschuwingen:
            print(f"  ⚠️  {w}")
    else:
        print("\n✓ Geen waarschuwingen")
    print()


def analyseer_relaties(systeem) -> Dict:
    """
    Analyseer de relationele structuur van het systeem (v4-specifiek).

    Returns
    -------
    dict
        Analyse van α-matrix met interpretatie
    """
    alpha = systeem.alpha_matrix
    actoren = [a.naam for a in systeem.actoren]
    n = len(actoren)

    relaties = []
    for i in range(n):
        for j in range(n):
            if i != j and alpha[i, j] != 0:
                richting = "wil naar" if alpha[i, j] > 0 else "wil weg van"
                sterkte = "sterk" if abs(alpha[i, j]) > 0.1 else "zwak"
                relaties.append({
                    'van': actoren[i],
                    'naar': actoren[j],
                    'alpha': alpha[i, j],
                    'richting': richting,
                    'sterkte': sterkte
                })

    # Wederkerigheid check
    wederkerig = []
    eenzijdig = []
    for i in range(n):
        for j in range(i+1, n):
            a_ij = alpha[i, j]
            a_ji = alpha[j, i]
            if a_ij != 0 or a_ji != 0:
                if np.sign(a_ij) == np.sign(a_ji) and a_ij != 0 and a_ji != 0:
                    wederkerig.append((actoren[i], actoren[j]))
                else:
                    eenzijdig.append((actoren[i], actoren[j]))

    return {
        'relaties': relaties,
        'wederkerig': wederkerig,
        'eenzijdig': eenzijdig,
        'gemiddelde_alpha': np.mean(alpha[alpha != 0]) if np.any(alpha != 0) else 0
    }
