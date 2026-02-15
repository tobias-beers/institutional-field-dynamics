"""
Institutional Field Dynamics v4 — Analysis Tools

Core question: "Does the configuration produce cooperation or conflict?"
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Diagnose:
    """Diagnostic result for an institutional force field (v4)."""
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
    samenwerking_score: float


def diagnose(systeem, d_gewenst: Optional[np.ndarray] = None,
             test_attractoren: bool = True) -> Diagnose:
    """
    Run diagnostics on the system (v4).

    Parameters
    ----------
    systeem : Krachtenveld
        The system to analyze
    d_gewenst : np.ndarray, optional
        Desired direction. Default: [-1, 1, 1] (lower workload, higher autonomy, more resources)
    test_attractoren : bool
        Whether to run attractor analysis (can be slow)

    Returns
    -------
    Diagnose
        Diagnostic object with all metrics
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

    # Attractor analysis
    n_attractoren = 0
    if test_attractoren:
        attractoren = systeem.vind_attractoren_multi(n_starts=10)
        n_attractoren = len(attractoren)

    # Cooperation score: mean α
    alpha_flat = systeem.alpha_matrix.flatten()
    alpha_flat = alpha_flat[alpha_flat != 0]
    samenwerking_score = np.mean(alpha_flat) if len(alpha_flat) > 0 else 0.0

    # Generate warnings
    waarschuwingen = []

    if asym > 0.8:
        waarschuwingen.append(
            f"HIGH ASYMMETRY W ({asym:.2f}): Power structure is strongly unilateral."
        )

    if samenwerking_score < -0.03:
        waarschuwingen.append(
            f"CONFLICT DETECTED: Mean α is negative ({samenwerking_score:.3f}). "
            "Actors push away from each other → conflict dynamics."
        )
    elif samenwerking_score > 0.05:
        pass

    if n_attractoren > 3:
        waarschuwingen.append(
            f"MULTIPLE ATTRACTORS ({n_attractoren}): System can converge to different "
            "equilibria → outcome depends on initial conditions."
        )

    if coh > 0.7 and eff_coh < 0.3:
        waarschuwingen.append(
            f"COHERENCE IN WRONG DIRECTION: Forces are coherent ({coh:.2f}) but "
            f"not toward the desired direction (eff. coh. = {eff_coh:.2f})."
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
    Compare system before and after policy intervention (v4).
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
        interpretatie.append("COOPERATION INCREASED: α has become more positive")
    elif delta['samenwerking_score'] < -0.02:
        interpretatie.append("CONFLICT INCREASED: α has become more negative")

    if delta['n_attractoren'] < 0:
        interpretatie.append(f"STABILITY IMPROVED: Fewer attractors ({diag_voor.n_attractoren} → {diag_na.n_attractoren})")
    elif delta['n_attractoren'] > 0:
        interpretatie.append(f"STABILITY DEGRADED: More attractors ({diag_voor.n_attractoren} → {diag_na.n_attractoren})")

    if delta['asymmetrie_W'] > 0.1:
        interpretatie.append("W has become MORE ASYMMETRIC")
    elif delta['asymmetrie_W'] < -0.1:
        interpretatie.append("W has become MORE SYMMETRIC (more reciprocal influence)")

    return {
        'voor': diag_voor,
        'na': diag_na,
        'delta': delta,
        'interpretatie': interpretatie
    }


def print_diagnose(diag: Diagnose, titel: str = "DIAGNOSTIC v4"):
    """Print diagnostic in readable format."""
    print("=" * 60)
    print(titel)
    print("=" * 60)
    print(f"\nAsymmetry W:           {diag.asymmetrie_W:.3f}")
    print(f"Coherence:             {diag.coherentie:.3f}")
    print(f"Effective coherence:   {diag.effectieve_coherentie:.3f}")
    print(f"Energy:                {diag.energie:.3f}")
    print(f"Cooperation score (α): {diag.samenwerking_score:.3f}")
    print(f"Number of attractors:  {diag.n_attractoren}")

    print("\nPOSITIONS:")
    for naam, pos in diag.posities.items():
        print(f"  {naam}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    if diag.waarschuwingen:
        print("\nWARNINGS:")
        for w in diag.waarschuwingen:
            print(f"  !! {w}")
    else:
        print("\nNo warnings")
    print()


def analyseer_relaties(systeem) -> Dict:
    """
    Analyze the relational structure of the system (v4-specific).

    Returns
    -------
    dict
        Analysis of α-matrix with interpretation
    """
    alpha = systeem.alpha_matrix
    actoren = [a.naam for a in systeem.actoren]
    n = len(actoren)

    relaties = []
    for i in range(n):
        for j in range(n):
            if i != j and alpha[i, j] != 0:
                direction = "attracted to" if alpha[i, j] > 0 else "repelled by"
                strength = "strong" if abs(alpha[i, j]) > 0.1 else "weak"
                relaties.append({
                    'from': actoren[i],
                    'to': actoren[j],
                    'alpha': alpha[i, j],
                    'direction': direction,
                    'strength': strength
                })

    # Reciprocity check
    reciprocal = []
    unilateral = []
    for i in range(n):
        for j in range(i+1, n):
            a_ij = alpha[i, j]
            a_ji = alpha[j, i]
            if a_ij != 0 or a_ji != 0:
                if np.sign(a_ij) == np.sign(a_ji) and a_ij != 0 and a_ji != 0:
                    reciprocal.append((actoren[i], actoren[j]))
                else:
                    unilateral.append((actoren[i], actoren[j]))

    return {
        'relations': relaties,
        'reciprocal': reciprocal,
        'unilateral': unilateral,
        'mean_alpha': np.mean(alpha[alpha != 0]) if np.any(alpha != 0) else 0
    }
