"""
Institutional Field Dynamics v4: Relational, Dynamic, Attractor-Aware

Key extensions over v3:
1. RELATIONAL U: Utility function depends on relations with others
2. DYNAMIC W: Power matrix evolves during simulation
3. ATTRACTOR ANALYSIS: Explicit test of multiple initializations
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class Actor:
    """An actor in the institutional force field."""
    naam: str
    positie: np.ndarray
    kleur: str = '#666666'

    def copy(self):
        return Actor(self.naam, self.positie.copy(), self.kleur)


class Krachtenveld:
    """
    Institutional dynamics as a force field (v4): Relational, Dynamic, Attractor-Aware.

    KEY EXTENSIONS OVER V3:

    1. RELATIONAL U: Utility function depends on relations with others
       U_i = -½g_i||x_i - d_i||² - ½Σ_k α_ik||x_i - x_k||²

    2. DYNAMIC W: Power matrix evolves during simulation
       W_ik(t+1) = W_ik(t) + η·alignment(F_i, F_k)

    3. ATTRACTOR ANALYSIS: Explicit test of multiple initializations

    Parameters
    ----------
    actoren : List[Actor]
        List of actors
    U_config : dict
        Utility configuration per actor:
        - 'doel': target ideal point
        - 'gewicht': goal weight (intrinsic motivation strength)
        - 'alpha': dict of relational coefficients toward other actors
          (positive = cooperation/proximity, negative = conflict/distance)
    W : np.ndarray
        Initial power matrix. W[i,j] = influence of j on i.
    C : dict, optional
        Constraints per actor
    W_fixed : np.ndarray, optional
        Mask for fixed W elements (1=fixed, 0=dynamic)
    eta : float
        Learning rate for W-evolution (0 = static W as in v3)
    alpha : float
        Step size for position update
    ruis : float
        Noise in the dynamics
    """

    def __init__(self, actoren: List[Actor],
                 U_config: Dict[str, dict],
                 W: np.ndarray,
                 C: Optional[Dict[str, dict]] = None,
                 W_fixed: Optional[np.ndarray] = None,
                 eta: float = 0.01,
                 alpha: float = 0.08,
                 ruis: float = 0.01):

        self.actoren = actoren
        self.U_config = U_config
        self.W = W.astype(float)
        self.W_fixed = W_fixed if W_fixed is not None else np.zeros_like(W)
        self.C = C or {}
        self.eta = eta
        self.alpha = alpha
        self.ruis = ruis
        self.n_dims = len(actoren[0].positie)
        self.n_actoren = len(actoren)

        self._build_alpha_matrix()

        # History
        self.geschiedenis = [[a.positie.copy() for a in actoren]]
        self.kracht_hist = []
        self.coherentie_hist = []
        self.energie_hist = []
        self.W_hist = [W.copy()]

    def _build_alpha_matrix(self):
        """Build the alpha matrix (relational component) from U_config."""
        self.alpha_matrix = np.zeros((self.n_actoren, self.n_actoren))
        for i, actor in enumerate(self.actoren):
            config = self.U_config.get(actor.naam, {})
            alpha_dict = config.get('alpha', {})
            for j, andere in enumerate(self.actoren):
                if andere.naam in alpha_dict:
                    self.alpha_matrix[i, j] = alpha_dict[andere.naam]

    def kracht(self, i: int) -> np.ndarray:
        """
        Force on actor i = gradient of RELATIONAL utility function.

        F_i = g_i(d_i - x_i)              [intrinsic: pull toward ideal]
            + Σ_k α_ik(x_k - x_i)         [relational force]
            + Σ_k W_ik(x_k - x_i)         [institutional force]
        """
        actor = self.actoren[i]
        config = self.U_config.get(actor.naam, {})

        # Intrinsic force: pull toward ideal
        doel = config.get('doel', actor.positie)
        gewicht = config.get('gewicht', 1.0)
        F_doel = gewicht * (doel - actor.positie)

        # Relational force (α)
        F_relationeel = np.zeros(self.n_dims)
        for j, andere in enumerate(self.actoren):
            if i != j:
                F_relationeel += self.alpha_matrix[i, j] * (andere.positie - actor.positie)

        # Institutional force (W)
        F_institutioneel = np.zeros(self.n_dims)
        for j, andere in enumerate(self.actoren):
            if i != j:
                F_institutioneel += self.W[i, j] * (andere.positie - actor.positie)

        return F_doel + F_relationeel + F_institutioneel

    def alle_krachten(self) -> List[np.ndarray]:
        return [self.kracht(i) for i in range(self.n_actoren)]

    def update_W(self, krachten: List[np.ndarray]):
        """
        Update W based on observed force alignment (dynamic W-evolution).

        Rule: when forces are aligned, mutual influence grows.
        """
        if self.eta <= 0:
            return

        for i in range(self.n_actoren):
            for j in range(self.n_actoren):
                if i != j and self.W_fixed[i, j] == 0:
                    Fi_norm = np.linalg.norm(krachten[i])
                    Fj_norm = np.linalg.norm(krachten[j])

                    if Fi_norm > 0.01 and Fj_norm > 0.01:
                        alignment = np.dot(krachten[i], krachten[j]) / (Fi_norm * Fj_norm)
                        delta_W = self.eta * alignment
                        self.W[i, j] = np.clip(self.W[i, j] + delta_W, -1.0, 2.0)

    def coherentie(self) -> float:
        """Measure force coherence (value-free: direction-agnostic)."""
        krachten = self.alle_krachten()
        F_totaal = np.sum(krachten, axis=0)

        norm_totaal_sq = np.sum(F_totaal**2)
        som_normen_sq = sum(np.sum(F**2) for F in krachten)

        if som_normen_sq < 1e-10:
            return 1.0

        return norm_totaal_sq / som_normen_sq

    def effectieve_coherentie(self, d: np.ndarray) -> float:
        """Measure coherence PROJECTED onto desired direction d."""
        krachten = self.alle_krachten()
        F_totaal = np.sum(krachten, axis=0)

        som_normen_sq = sum(np.sum(F**2) for F in krachten)
        d_norm_sq = np.sum(d**2)

        if som_normen_sq < 1e-10 or d_norm_sq < 1e-10:
            return 0.0

        projectie = np.dot(F_totaal, d)
        return (projectie ** 2) / (som_normen_sq * d_norm_sq)

    def energie(self) -> float:
        """Total energy in the system."""
        krachten = self.alle_krachten()
        return sum(np.linalg.norm(F) for F in krachten)

    def constructieve_spanning(self, d: Optional[np.ndarray] = None) -> float:
        """Constructive tension: energy × effective coherence."""
        if d is not None:
            return self.energie() * self.effectieve_coherentie(d)
        return self.energie() * self.coherentie()

    def asymmetrie_W(self) -> float:
        """Compute power asymmetry index A(W)."""
        W_norm = np.linalg.norm(self.W)
        if W_norm < 1e-10:
            return 0.0
        return np.linalg.norm(self.W - self.W.T) / W_norm

    def vind_attractor(self, max_stappen: int = 500, threshold: float = 0.01) -> np.ndarray:
        """Find the attractor from current position."""
        origineel = [a.positie.copy() for a in self.actoren]
        W_origineel = self.W.copy()

        for _ in range(max_stappen):
            oude_posities = [a.positie.copy() for a in self.actoren]
            self._stap_zonder_history()

            delta = max(np.linalg.norm(self.actoren[i].positie - oude_posities[i])
                       for i in range(self.n_actoren))
            if delta < threshold:
                break

        attractor = np.array([a.positie.copy() for a in self.actoren])

        for i, a in enumerate(self.actoren):
            a.positie = origineel[i]
        self.W = W_origineel

        return attractor

    def vind_attractoren_multi(self, n_starts: int = 10,
                                bereik: Tuple[float, float] = (2.0, 8.0),
                                max_stappen: int = 500) -> List[np.ndarray]:
        """
        Find attractors from multiple random starting points.

        Essential for attractor analysis:
        - If all starts converge to the same attractor → single stable equilibrium
        - If starts converge to different attractors → multi-stability
        """
        origineel = [a.positie.copy() for a in self.actoren]
        W_origineel = self.W.copy()

        attractoren = []

        for _ in range(n_starts):
            for actor in self.actoren:
                actor.positie = np.random.uniform(bereik[0], bereik[1], self.n_dims)
            self.W = W_origineel.copy()

            attractor = self.vind_attractor(max_stappen=max_stappen)

            is_nieuw = True
            for bestaand in attractoren:
                if np.allclose(attractor, bestaand, atol=0.3):
                    is_nieuw = False
                    break

            if is_nieuw:
                attractoren.append(attractor)

        for i, a in enumerate(self.actoren):
            a.positie = origineel[i]
        self.W = W_origineel

        return attractoren

    def _stap_zonder_history(self):
        """Internal step without history update."""
        krachten = self.alle_krachten()
        self.update_W(krachten)
        for i, actor in enumerate(self.actoren):
            actor.positie = actor.positie + self.alpha * krachten[i]
            self._pas_constraints_toe(actor)

    def _pas_constraints_toe(self, actor: Actor):
        """Apply constraints to one actor."""
        if actor.naam in self.C:
            c = self.C[actor.naam]
            for dim, (lo, hi) in c.items():
                actor.positie[dim] = np.clip(actor.positie[dim], lo, hi)

    def transformeer(self,
                     nieuwe_W: Optional[np.ndarray] = None,
                     nieuwe_U_config: Optional[Dict[str, dict]] = None,
                     nieuwe_C: Optional[Dict[str, dict]] = None):
        """
        POLICY = TRANSFORMATION OF THE FIELD.

        P: (U, W, C, α) → (U', W', C', α')
        """
        if nieuwe_W is not None:
            self.W = nieuwe_W.astype(float)
        if nieuwe_U_config is not None:
            for naam, config in nieuwe_U_config.items():
                if naam in self.U_config:
                    self.U_config[naam].update(config)
                else:
                    self.U_config[naam] = config
            self._build_alpha_matrix()
        if nieuwe_C is not None:
            self.C.update(nieuwe_C)

    def stap(self):
        """One time step with dynamic W-evolution."""
        krachten = self.alle_krachten()
        self.update_W(krachten)

        for i, actor in enumerate(self.actoren):
            noise = np.random.normal(0, self.ruis, self.n_dims)
            actor.positie = actor.positie + self.alpha * krachten[i] + noise
            self._pas_constraints_toe(actor)

        self.geschiedenis.append([a.positie.copy() for a in self.actoren])
        self.kracht_hist.append([k.copy() for k in krachten])
        self.coherentie_hist.append(self.coherentie())
        self.energie_hist.append(self.energie())
        self.W_hist.append(self.W.copy())

    def simuleer(self, n: int):
        for _ in range(n):
            self.stap()
        return np.array(self.geschiedenis)

    def reset(self):
        """Reset history."""
        self.geschiedenis = [[a.positie.copy() for a in self.actoren]]
        self.kracht_hist = []
        self.coherentie_hist = []
        self.energie_hist = []
        self.W_hist = [self.W.copy()]
