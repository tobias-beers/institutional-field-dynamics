"""
Institutional Field Dynamics v4 — Visualization Tools
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_simulatie(systeem, titel: str, t_beleid: Optional[int] = None,
                   labels: List[str] = ['Workload', 'Autonomy', 'Resources']):
    """
    Plot simulation with v4 features including W-evolution.
    """
    hist = np.array(systeem.geschiedenis)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Row 1: Positions per dimension
    for d, (ax, label) in enumerate(zip(axes[0], labels)):
        for i, actor in enumerate(systeem.actoren):
            ax.plot(hist[:, i, d], color=actor.kleur, linewidth=2, label=actor.naam)
            config = systeem.U_config.get(actor.naam, {})
            if 'doel' in config:
                ax.axhline(y=config['doel'][d], color=actor.kleur, linestyle=':', alpha=0.5)
        if t_beleid:
            ax.axvline(x=t_beleid, color='red', linestyle='--', linewidth=2)
        ax.set_ylabel(label)
        ax.set_title(f'{label}', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 10)
        if d == 0:
            ax.legend(fontsize=8)

    # Row 2: System metrics
    ax = axes[1, 0]
    if systeem.coherentie_hist:
        ax.plot(systeem.coherentie_hist, color='#2ecc71', linewidth=2)
        ax.fill_between(range(len(systeem.coherentie_hist)),
                        systeem.coherentie_hist, alpha=0.3, color='#2ecc71')
    if t_beleid:
        ax.axvline(x=t_beleid, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Coherence')
    ax.set_xlabel('Time')
    ax.set_title('Coherence', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if systeem.energie_hist:
        ax.plot(systeem.energie_hist, color='#e74c3c', linewidth=2)
        ax.fill_between(range(len(systeem.energie_hist)),
                        systeem.energie_hist, alpha=0.3, color='#e74c3c')
    if t_beleid:
        ax.axvline(x=t_beleid, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Energy')
    ax.set_xlabel('Time')
    ax.set_title('Energy', fontweight='bold')
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    if systeem.coherentie_hist and systeem.energie_hist:
        constr_span = [e * c for e, c in zip(systeem.energie_hist, systeem.coherentie_hist)]
        ax.plot(constr_span, color='#9b59b6', linewidth=2)
        ax.fill_between(range(len(constr_span)), constr_span, alpha=0.3, color='#9b59b6')
    if t_beleid:
        ax.axvline(x=t_beleid, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Constructive Tension')
    ax.set_xlabel('Time')
    ax.set_title('Energy × Coherence', fontweight='bold')
    ax.grid(alpha=0.3)

    # Row 3: W-evolution
    ax = axes[2, 0]
    if systeem.W_hist and len(systeem.W_hist) > 1:
        W_hist = np.array(systeem.W_hist)
        n_actors = W_hist.shape[1]
        colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
        for i in range(min(n_actors, 3)):
            for j in range(min(n_actors, 3)):
                if i != j:
                    ax.plot(W_hist[:, i, j], linewidth=1.5, alpha=0.7,
                           label=f'W[{i},{j}]', color=colors[(i*n_actors+j) % len(colors)])
    if t_beleid:
        ax.axvline(x=t_beleid, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('W value')
    ax.set_xlabel('Time')
    ax.set_title('W-evolution', fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)

    # W asymmetry over time
    ax = axes[2, 1]
    if systeem.W_hist and len(systeem.W_hist) > 1:
        W_hist = np.array(systeem.W_hist)
        asym = []
        for W_t in W_hist:
            W_norm = np.linalg.norm(W_t)
            if W_norm > 1e-10:
                asym.append(np.linalg.norm(W_t - W_t.T) / W_norm)
            else:
                asym.append(0)
        ax.plot(asym, color='#8e44ad', linewidth=2)
        ax.fill_between(range(len(asym)), asym, alpha=0.3, color='#8e44ad')
    if t_beleid:
        ax.axvline(x=t_beleid, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Asymmetry')
    ax.set_xlabel('Time')
    ax.set_title('W Asymmetry Over Time', fontweight='bold')
    ax.grid(alpha=0.3)

    # Total W
    ax = axes[2, 2]
    if systeem.W_hist and len(systeem.W_hist) > 1:
        W_hist = np.array(systeem.W_hist)
        W_sum = [np.sum(W_t) for W_t in W_hist]
        ax.plot(W_sum, color='#16a085', linewidth=2)
        ax.fill_between(range(len(W_sum)), W_sum, alpha=0.3, color='#16a085')
    if t_beleid:
        ax.axvline(x=t_beleid, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Sum W')
    ax.set_xlabel('Time')
    ax.set_title('Total Influence', fontweight='bold')
    ax.grid(alpha=0.3)

    fig.suptitle(titel, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_attractoren(systeem, attractoren: List[np.ndarray], titel: str = "Attractor Analysis"):
    """
    Visualize discovered attractors (v4-specific).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, attr in enumerate(attractoren):
        for i, actor in enumerate(systeem.actoren):
            marker = 'o' if idx == 0 else ['s', '^', 'D', 'v', 'p'][idx % 5]
            ax.scatter(attr[i, 0], attr[i, 1], s=150,
                      marker=marker,
                      c=actor.kleur, edgecolors='black', linewidth=2,
                      alpha=0.7, label=f'{actor.naam} (attr {idx+1})' if i == 0 else '')

    ax.set_xlabel('Workload')
    ax.set_ylabel('Autonomy')
    ax.set_title(f'{titel}: {len(attractoren)} attractor(s) found', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    return fig


def plot_relatienetwerk(systeem, titel: str = "Relation Network"):
    """
    Visualize the α-matrix as a network (v4-specific).
    """
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 8))

    n = len(systeem.actoren)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 3

    # Actor positions in circle
    pos = {}
    for i, actor in enumerate(systeem.actoren):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        pos[actor.naam] = (x, y)

        ax.scatter(x, y, s=500, c=actor.kleur, edgecolors='black', linewidth=2, zorder=10)
        ax.annotate(actor.naam, (x, y), ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw relations (α)
    for i, actor_i in enumerate(systeem.actoren):
        for j, actor_j in enumerate(systeem.actoren):
            if i != j:
                alpha = systeem.alpha_matrix[i, j]
                if abs(alpha) > 0.01:
                    x1, y1 = pos[actor_i.naam]
                    x2, y2 = pos[actor_j.naam]

                    color = '#27ae60' if alpha > 0 else '#e74c3c'
                    width = min(abs(alpha) * 20, 3)

                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle='->', color=color, lw=width, alpha=0.6))

    # Legend
    pos_patch = mpatches.Patch(color='#27ae60', label='Cooperation (α > 0)')
    neg_patch = mpatches.Patch(color='#e74c3c', label='Conflict (α < 0)')
    ax.legend(handles=[pos_patch, neg_patch], loc='upper right')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(titel, fontsize=14, fontweight='bold')

    return fig
