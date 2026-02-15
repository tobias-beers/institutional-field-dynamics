"""
Advisory Drift Book — Typology & Drift Simulations
Must-haves: 3.2 (four typologies), 3.3 (drift mechanism), 3.5 (hysteresis)
Nice-to-haves: 3.4 (AADD intervention), 3.6 (socialization)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from beleidsdynamica_v4 import Krachtenveld, Actor
from beleidsdynamica_v4.analyse import diagnose

np.random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)


# ═══════════════════════════════════════════════════════
# Advisory Drift configurations
#
# Two core actors: Decision-Maker (DM) and Advisory Body (AB)
# Three dimensions: Decision Closure, Agency Allocation, Trust
#   dim 0 = closure / lock-in (high = decision committed)
#   dim 1 = agency / influence (high = real design authority)
#   dim 2 = relationship quality (high = trust, low = conflict)
# ═══════════════════════════════════════════════════════════


def make_advisory_system(C_level, G_level, alpha_ab_dm, alpha_dm_ab,
                         ab_goal_agency=8.0, dm_goal_closure=8.0,
                         W_dm_to_ab=0.4, W_ab_to_dm=0.1,
                         eta=0.02, n_extra_members=0):
    """Create an advisory drift configuration.

    Parameters
    ----------
    C_level : float
        Decision closure (DM starting position on dim 0). High = locked in.
    G_level : float
        Agency allocation (AB starting position on dim 1). High = real design authority.
    alpha_ab_dm : float
        AB's relational coefficient toward DM. Negative = conflict.
    alpha_dm_ab : float
        DM's relational coefficient toward AB. Negative = defensive.
    ab_goal_agency : float
        AB's ideal on agency dimension.
    dm_goal_closure : float
        DM's ideal on closure dimension.
    W_dm_to_ab : float
        DM's institutional influence on AB.
    W_ab_to_dm : float
        AB's institutional influence on DM (typically lower = asymmetric).
    eta : float
        W-evolution learning rate.
    n_extra_members : int
        Extra advisory body members (for socialization experiments).
    """
    actoren = [
        Actor("Decision-Maker", np.array([C_level, 3.0, 6.0]), '#e74c3c'),
        Actor("Advisory Body", np.array([4.0, G_level, 6.0]), '#2c3e50'),
    ]

    # Add extra members if requested
    for i in range(n_extra_members):
        pos = np.array([4.0, G_level + np.random.uniform(-0.5, 0.5),
                        6.0 + np.random.uniform(-0.5, 0.5)])
        actoren.append(Actor(f"Member-{i+1}", pos, '#7f8c8d'))

    n = len(actoren)

    U_config = {
        "Decision-Maker": {
            'doel': np.array([dm_goal_closure, 3.0, 7.0]),
            'gewicht': 0.5,
            'alpha': {'Advisory Body': alpha_dm_ab},
        },
        "Advisory Body": {
            'doel': np.array([3.0, ab_goal_agency, 8.0]),
            'gewicht': 0.4,
            'alpha': {'Decision-Maker': alpha_ab_dm},
        },
    }

    # Extra members: same goals as AB, inherit alpha
    for i in range(n_extra_members):
        name = f"Member-{i+1}"
        U_config[name] = {
            'doel': np.array([3.0, ab_goal_agency + np.random.uniform(-0.5, 0.5), 8.0]),
            'gewicht': 0.35,
            'alpha': {'Decision-Maker': alpha_ab_dm * 0.8,
                      'Advisory Body': 0.05},
        }
        # AB also relates to each member
        U_config["Advisory Body"]['alpha'][name] = 0.03
        U_config["Decision-Maker"]['alpha'][name] = alpha_dm_ab * 0.5

    # W matrix: DM influences AB more than reverse
    W = np.zeros((n, n))
    W[1, 0] = W_dm_to_ab  # DM → AB
    W[0, 1] = W_ab_to_dm  # AB → DM
    for i in range(2, n):
        W[i, 0] = W_dm_to_ab * 0.5  # DM → members
        W[0, i] = W_ab_to_dm * 0.3  # members → DM
        W[i, 1] = 0.2               # AB → members (peer influence)
        W[1, i] = 0.1               # members → AB

    C = {a.naam: {0: (0, 10), 1: (0, 10), 2: (0, 10)} for a in actoren}

    return Krachtenveld(actoren, U_config, W, C=C, eta=eta)


# ═══════════════════════════════════════════════════
# EXPERIMENT 3.2: Four Typologies Simulation
# ═══════════════════════════════════════════════════

def experiment_four_typologies():
    print("=" * 60)
    print("EXPERIMENT 3.2: Four Typologies Simulation")
    print("=" * 60)

    configs = {
        'Type I\nOpen Co-Design': {
            'C_level': 3.0, 'G_level': 7.0,
            'alpha_ab_dm': 0.05, 'alpha_dm_ab': 0.03,
            'W_dm_to_ab': 0.2, 'W_ab_to_dm': 0.15,
        },
        'Type II\nGuided Design': {
            'C_level': 7.0, 'G_level': 7.0,
            'alpha_ab_dm': 0.02, 'alpha_dm_ab': 0.01,
            'W_dm_to_ab': 0.3, 'W_ab_to_dm': 0.1,
        },
        'Type III\nCorrective Open': {
            'C_level': 3.0, 'G_level': 3.0,
            'alpha_ab_dm': -0.02, 'alpha_dm_ab': -0.01,
            'W_dm_to_ab': 0.4, 'W_ab_to_dm': 0.05,
        },
        'Type IV\nMandatory Corrective': {
            'C_level': 8.0, 'G_level': 2.0,
            'alpha_ab_dm': -0.05, 'alpha_dm_ab': -0.03,
            'W_dm_to_ab': 0.5, 'W_ab_to_dm': 0.05,
        },
    }

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    results = {}
    d = np.array([-1, 1, 1]) / np.sqrt(3)  # desired: less closure, more agency, more trust

    for idx, (name, cfg) in enumerate(configs.items()):
        sys = make_advisory_system(**cfg)
        attractors = sys.vind_attractoren_multi(n_starts=15)
        n_att = len(attractors)
        diag = diagnose(sys, d_gewenst=d, test_attractoren=False)

        # Run simulation for trajectory
        sys2 = make_advisory_system(**cfg)
        history = sys2.simuleer(300)

        # Extract trajectories
        dm_traj = history[:, 0, :]  # Decision-Maker
        ab_traj = history[:, 1, :]  # Advisory Body

        results[name] = {
            'n_attractoren': n_att,
            'eff_coherentie': diag.effectieve_coherentie,
            'asymmetrie_W': diag.asymmetrie_W,
            'samenwerking': diag.samenwerking_score,
            'dm_final': dm_traj[-1],
            'ab_final': ab_traj[-1],
        }

        # Subplot: trajectory in Closure × Agency space
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        # Dim 0 = closure, dim 1 = agency
        ax.plot(dm_traj[:, 0], dm_traj[:, 1], '-', color='#e74c3c', alpha=0.7, linewidth=1.5)
        ax.plot(ab_traj[:, 0], ab_traj[:, 1], '-', color='#2c3e50', alpha=0.7, linewidth=1.5)

        # Start points
        ax.plot(dm_traj[0, 0], dm_traj[0, 1], 'o', color='#e74c3c', markersize=10, zorder=5)
        ax.plot(ab_traj[0, 0], ab_traj[0, 1], 'o', color='#2c3e50', markersize=10, zorder=5)

        # End points
        ax.plot(dm_traj[-1, 0], dm_traj[-1, 1], 's', color='#e74c3c', markersize=12, zorder=5, label='DM')
        ax.plot(ab_traj[-1, 0], ab_traj[-1, 1], 's', color='#2c3e50', markersize=12, zorder=5, label='AB')

        ax.set_xlabel('Decision Closure')
        ax.set_ylabel('Agency Allocation')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title(f'{name}\nAttractors: {n_att} | ECoh: {diag.effectieve_coherentie:.2f} | A(W): {diag.asymmetrie_W:.2f}',
                     fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Advisory Drift: Four Typologies in Configuration Space', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUT, 'four_typologies.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Summary table
    print(f"\n{'Type':>30} | {'Attractors':>10} | {'Eff.Coh':>8} | {'A(W)':>8} | {'Coop':>8}")
    print("-" * 80)
    for name, r in results.items():
        name_clean = name.replace('\n', ' — ')
        print(f"{name_clean:>30} | {r['n_attractoren']:>10} | {r['eff_coherentie']:>8.3f} | {r['asymmetrie_W']:>8.3f} | {r['samenwerking']:>8.3f}")

    print(f"\nFigure saved: {OUT}/four_typologies.png")
    return results


# ═══════════════════════════════════════════════════
# EXPERIMENT 3.3: Drift Mechanism Simulation
# ═══════════════════════════════════════════════════

def experiment_drift_mechanism():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3.3: Drift Mechanism — The Movie")
    print("=" * 60)

    # Start from Type II (reasonable starting point: committed decision, real design space)
    # Then progressively remove agency and increase closure → watch drift emerge
    sys = make_advisory_system(
        C_level=6.0, G_level=6.0,        # Moderate closure, moderate agency
        alpha_ab_dm=0.02, alpha_dm_ab=0.01,  # Starts cooperative
        W_dm_to_ab=0.3, W_ab_to_dm=0.10,
        eta=0.03,  # Dynamic W evolution
    )

    n_steps = 400
    # Intervention schedule: gradually increase closure, decrease agency
    # This simulates the progressive narrowing of advisory space

    dm_positions = []
    ab_positions = []
    coherence_hist = []
    asymmetry_hist = []
    alpha_ab_dm_hist = []
    friction_hist = []

    d = np.array([-1, 1, 1]) / np.sqrt(3)

    for t in range(n_steps):
        dm_positions.append(sys.actoren[0].positie.copy())
        ab_positions.append(sys.actoren[1].positie.copy())
        coherence_hist.append(sys.effectieve_coherentie(d))
        asymmetry_hist.append(sys.asymmetrie_W())
        alpha_ab_dm_hist.append(sys.alpha_matrix[1, 0])

        # Compute friction (opposing forces)
        F_dm = sys.kracht(0)
        F_ab = sys.kracht(1)
        friction = max(0, -np.dot(F_dm, F_ab))
        friction_hist.append(friction)

        sys.stap()

        # Phase transitions in the structural parameters
        # Phase 1 (t=0-80): Engagement — advisory body explores design space
        # Phase 2 (t=80-160): Narrowing — DM starts constraining agency
        # Phase 3 (t=160-240): Conflict — AB's domain search meets resistance
        # Phase 4 (t=240-320): Hardening — alpha goes negative, identity forms
        # Phase 5 (t=320-400): Stabilized opposition

        if t == 80:
            # DM narrows the advisory domain: increase closure pressure
            sys.U_config["Decision-Maker"]['doel'][0] = 9.0  # stronger closure goal
            sys.U_config["Decision-Maker"]['gewicht'] = 0.6
            sys._build_alpha_matrix()
            print(f"  t={t}: DM increases closure pressure (doel_closure → 9.0)")

        if t == 160:
            # Conflict emerges: alpha becomes negative
            sys.U_config["Advisory Body"]['alpha']['Decision-Maker'] = -0.03
            sys.U_config["Decision-Maker"]['alpha']['Advisory Body'] = -0.02
            sys._build_alpha_matrix()
            print(f"  t={t}: Conflict emerges (alpha_AB→DM = -0.03)")

        if t == 240:
            # Identity crystallization: alpha deepens, AB goal hardens
            sys.U_config["Advisory Body"]['alpha']['Decision-Maker'] = -0.06
            sys.U_config["Decision-Maker"]['alpha']['Advisory Body'] = -0.04
            sys.U_config["Advisory Body"]['doel'] = np.array([2.0, 9.0, 3.0])  # opposition ideal
            sys.U_config["Advisory Body"]['gewicht'] = 0.5
            sys._build_alpha_matrix()
            print(f"  t={t}: Identity crystallization (alpha_AB→DM = -0.06, AB goal hardened)")

    dm_arr = np.array(dm_positions)
    ab_arr = np.array(ab_positions)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Trajectory in Closure × Agency space
    ax = axes[0, 0]
    scatter_dm = ax.scatter(dm_arr[:, 0], dm_arr[:, 1], c=range(n_steps), cmap='Reds', s=3, alpha=0.7)
    scatter_ab = ax.scatter(ab_arr[:, 0], ab_arr[:, 1], c=range(n_steps), cmap='Blues', s=3, alpha=0.7)
    ax.plot(dm_arr[0, 0], dm_arr[0, 1], 'ro', markersize=10, label='DM start')
    ax.plot(ab_arr[0, 0], ab_arr[0, 1], 'bo', markersize=10, label='AB start')
    ax.plot(dm_arr[-1, 0], dm_arr[-1, 1], 'r^', markersize=12, label='DM end')
    ax.plot(ab_arr[-1, 0], ab_arr[-1, 1], 'b^', markersize=12, label='AB end')
    for t_mark in [80, 160, 240]:
        ax.axvline(x=dm_arr[t_mark, 0], color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Decision Closure')
    ax.set_ylabel('Agency Allocation')
    ax.set_title('Trajectory: Closure x Agency')
    ax.legend(fontsize=7)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.2)

    # 2. Positions over time
    ax = axes[0, 1]
    ax.plot(dm_arr[:, 0], color='#e74c3c', label='DM Closure', linewidth=1.5)
    ax.plot(ab_arr[:, 1], color='#2c3e50', label='AB Agency', linewidth=1.5)
    ax.plot(ab_arr[:, 2], color='#3498db', label='AB Trust', linewidth=1, alpha=0.7)
    for t_mark in [80, 160, 240]:
        ax.axvline(x=t_mark, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position')
    ax.set_title('Key Dimensions Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 3. Effective coherence over time
    ax = axes[0, 2]
    ax.plot(coherence_hist, color='#27ae60', linewidth=1.5)
    for t_mark in [80, 160, 240]:
        ax.axvline(x=t_mark, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Effective Coherence')
    ax.set_title('System Coherence Over Time')
    ax.grid(True, alpha=0.2)

    # 4. W Asymmetry over time
    ax = axes[1, 0]
    ax.plot(asymmetry_hist, color='#8e44ad', linewidth=1.5)
    for t_mark in [80, 160, 240]:
        ax.axvline(x=t_mark, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('W Asymmetry')
    ax.set_title('Power Asymmetry Evolution')
    ax.grid(True, alpha=0.2)

    # 5. Alpha (relational coefficient) over time
    ax = axes[1, 1]
    ax.plot(alpha_ab_dm_hist, color='#c0392b', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    for t_mark in [80, 160, 240]:
        ax.axvline(x=t_mark, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('alpha (AB toward DM)')
    ax.set_title('Relational Coefficient: Cooperation to Conflict')
    ax.grid(True, alpha=0.2)

    # 6. Friction over time
    ax = axes[1, 2]
    ax.plot(friction_hist, color='#d35400', linewidth=1, alpha=0.7)
    # Smoothed
    window = 10
    if len(friction_hist) > window:
        smoothed = np.convolve(friction_hist, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(friction_hist)), smoothed, color='#d35400', linewidth=2)
    for t_mark in [80, 160, 240]:
        ax.axvline(x=t_mark, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Friction')
    ax.set_title('Friction: Energy Wasted in Opposition')
    ax.grid(True, alpha=0.2)

    # Phase labels
    for ax_row in axes:
        for ax in ax_row:
            ylim = ax.get_ylim()
            y_top = ylim[1] - 0.05 * (ylim[1] - ylim[0])
            for t_start, t_end, label in [(0, 80, 'Engage'), (80, 160, 'Narrow'),
                                           (160, 240, 'Conflict'), (240, 400, 'Harden')]:
                ax.text((t_start + t_end) / 2, y_top, label, ha='center', fontsize=7,
                        alpha=0.4, style='italic')

    plt.suptitle('Advisory Drift: From Cooperation to Opposition\n(11-step mechanism as continuous simulation)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'drift_mechanism.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Summary
    print(f"\n  DM start:  [{dm_arr[0,0]:.1f}, {dm_arr[0,1]:.1f}, {dm_arr[0,2]:.1f}]")
    print(f"  DM end:    [{dm_arr[-1,0]:.1f}, {dm_arr[-1,1]:.1f}, {dm_arr[-1,2]:.1f}]")
    print(f"  AB start:  [{ab_arr[0,0]:.1f}, {ab_arr[0,1]:.1f}, {ab_arr[0,2]:.1f}]")
    print(f"  AB end:    [{ab_arr[-1,0]:.1f}, {ab_arr[-1,1]:.1f}, {ab_arr[-1,2]:.1f}]")
    print(f"  Final coherence: {coherence_hist[-1]:.3f}")
    print(f"  Final asymmetry: {asymmetry_hist[-1]:.3f}")
    print(f"  Final alpha (AB→DM): {alpha_ab_dm_hist[-1]:.3f}")
    print(f"  Final friction: {friction_hist[-1]:.3f}")
    print(f"\nFigure saved: {OUT}/drift_mechanism.png")

    return {
        'dm_trajectory': dm_arr,
        'ab_trajectory': ab_arr,
        'coherence': coherence_hist,
        'asymmetry': asymmetry_hist,
        'alpha': alpha_ab_dm_hist,
        'friction': friction_hist,
    }


# ═══════════════════════════════════════════════════
# EXPERIMENT 3.4: AADD Intervention Simulation
# ═══════════════════════════════════════════════════

def experiment_aadd_intervention():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3.4: AADD Intervention — Genuine vs Symbolic")
    print("=" * 60)

    d = np.array([-1, 1, 1]) / np.sqrt(3)

    # Start from Type IV oppositional configuration
    def make_type_iv():
        return make_advisory_system(
            C_level=8.0, G_level=2.0,
            alpha_ab_dm=-0.05, alpha_dm_ab=-0.03,
            W_dm_to_ab=0.5, W_ab_to_dm=0.05,
            eta=0.02,
        )

    # Baseline: no intervention
    sys_base = make_type_iv()
    sys_base.simuleer(200)
    diag_base = diagnose(sys_base, d_gewenst=d, test_attractoren=True)

    # Genuine AADD: increase AB agency on genuine open dimensions
    sys_genuine = make_type_iv()
    sys_genuine.simuleer(100)  # Let opposition establish
    # Intervention: increase AB agency, decrease closure, improve feedback
    sys_genuine.transformeer(
        nieuwe_U_config={
            "Advisory Body": {
                'doel': np.array([3.0, 8.0, 7.0]),
                'gewicht': 0.45,
                'alpha': {'Decision-Maker': 0.02},  # Reset to cooperative
            },
            "Decision-Maker": {
                'alpha': {'Advisory Body': 0.01},
            },
        },
    )
    # Also increase AB→DM influence (genuine feedback)
    sys_genuine.W[0, 1] = 0.15  # DM now listens to AB
    sys_genuine.simuleer(200)
    diag_genuine = diagnose(sys_genuine, d_gewenst=d, test_attractoren=True)

    # Symbolic AADD: "allocate domain" but keep it fake (constraints prevent real influence)
    sys_symbolic = make_type_iv()
    sys_symbolic.simuleer(100)  # Let opposition establish
    # Intervention: say you're giving agency, but keep W asymmetric and alpha negative
    sys_symbolic.transformeer(
        nieuwe_U_config={
            "Advisory Body": {
                'doel': np.array([3.0, 7.0, 6.0]),
                'gewicht': 0.4,
                # alpha stays negative — trust not restored
            },
        },
    )
    # W stays asymmetric — DM doesn't actually listen
    sys_symbolic.simuleer(200)
    diag_symbolic = diagnose(sys_symbolic, d_gewenst=d, test_attractoren=True)

    print(f"\n{'Scenario':>25} | {'Attractors':>10} | {'Eff.Coh':>8} | {'A(W)':>8} | {'Coop':>8}")
    print("-" * 75)
    for label, diag in [('No intervention', diag_base),
                        ('Genuine AADD', diag_genuine),
                        ('Symbolic AADD', diag_symbolic)]:
        print(f"{label:>25} | {diag.n_attractoren:>10} | {diag.effectieve_coherentie:>8.3f} | {diag.asymmetrie_W:>8.3f} | {diag.samenwerking_score:>8.3f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = ['No Intervention', 'Genuine AADD', 'Symbolic AADD']
    colors = ['#e74c3c', '#27ae60', '#f39c12']

    # Attractors
    ax = axes[0]
    vals = [diag_base.n_attractoren, diag_genuine.n_attractoren, diag_symbolic.n_attractoren]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel('Number of Attractors')
    ax.set_title('Structural Stability')

    # Effective coherence
    ax = axes[1]
    vals = [diag_base.effectieve_coherentie, diag_genuine.effectieve_coherentie, diag_symbolic.effectieve_coherentie]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel('Effective Coherence')
    ax.set_title('Alignment with Goal')

    # Cooperation score
    ax = axes[2]
    vals = [diag_base.samenwerking_score, diag_genuine.samenwerking_score, diag_symbolic.samenwerking_score]
    ax.bar(labels, vals, color=colors)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Cooperation Score (alpha)')
    ax.set_title('Relational Quality')

    plt.suptitle('AADD Intervention: Genuine Design Space vs Symbolic Allocation', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'aadd_intervention.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {OUT}/aadd_intervention.png")

    return {'baseline': diag_base, 'genuine': diag_genuine, 'symbolic': diag_symbolic}


# ═══════════════════════════════════════════════════
# EXPERIMENT 3.5: Helena Voss — Hysteresis
# ═══════════════════════════════════════════════════

def experiment_hysteresis():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3.5: Helena Voss — Hysteresis (Identity Persists)")
    print("=" * 60)

    d = np.array([-1, 1, 1]) / np.sqrt(3)

    n_steps_phase1 = 200  # Build up opposition
    n_steps_phase2 = 300  # Try to reform

    # Phase 1: Build oppositional attractor (Type IV → deep opposition)
    sys = make_advisory_system(
        C_level=8.0, G_level=2.0,
        alpha_ab_dm=-0.05, alpha_dm_ab=-0.03,
        W_dm_to_ab=0.5, W_ab_to_dm=0.05,
        eta=0.03,
    )

    # Track everything
    dm_positions = []
    ab_positions = []
    coherence_hist = []
    alpha_hist = []
    phase_labels = []

    # Phase 1: Opposition builds
    for t in range(n_steps_phase1):
        dm_positions.append(sys.actoren[0].positie.copy())
        ab_positions.append(sys.actoren[1].positie.copy())
        coherence_hist.append(sys.effectieve_coherentie(d))
        alpha_hist.append(sys.alpha_matrix[1, 0])
        phase_labels.append('Opposition')

        # Gradual identity deepening
        if t == 100:
            sys.U_config["Advisory Body"]['alpha']['Decision-Maker'] = -0.07
            sys.U_config["Decision-Maker"]['alpha']['Advisory Body'] = -0.05
            sys.U_config["Advisory Body"]['doel'] = np.array([2.0, 9.0, 3.0])
            sys._build_alpha_matrix()

        sys.stap()

    print(f"  After Phase 1 (opposition):")
    print(f"    AB position: [{sys.actoren[1].positie[0]:.1f}, {sys.actoren[1].positie[1]:.1f}, {sys.actoren[1].positie[2]:.1f}]")
    print(f"    alpha AB→DM: {sys.alpha_matrix[1,0]:.3f}")
    print(f"    Eff coherence: {coherence_hist[-1]:.3f}")

    # Phase 2: Helena Voss arrives — reforms the structure
    # Parameters improve to Type II / Type I levels
    # BUT alpha stays deeply negative (identity already formed)
    print(f"\n  t={n_steps_phase1}: Helena Voss arrives — structural reform begins")

    # Structural improvements
    sys.transformeer(
        nieuwe_U_config={
            "Decision-Maker": {
                'doel': np.array([5.0, 6.0, 7.0]),  # Less closure, more agency for AB
                'gewicht': 0.4,
                'alpha': {'Advisory Body': 0.03},  # Helena tries cooperation
            },
        },
    )
    # Improve W: make it more symmetric
    sys.W[0, 1] = 0.15  # DM now listens to AB
    sys.W[1, 0] = 0.25  # Reduce DM pressure on AB

    # KEY: Advisory body's alpha stays negative
    # This is the hysteresis — structure improved, but identity persists
    # Helena's reform is architecturally right but historically too late

    for t in range(n_steps_phase2):
        dm_positions.append(sys.actoren[0].positie.copy())
        ab_positions.append(sys.actoren[1].positie.copy())
        coherence_hist.append(sys.effectieve_coherentie(d))
        alpha_hist.append(sys.alpha_matrix[1, 0])
        phase_labels.append('Reform')
        sys.stap()

    dm_arr = np.array(dm_positions)
    ab_arr = np.array(ab_positions)

    print(f"\n  After Phase 2 (Helena Voss reform):")
    print(f"    AB position: [{sys.actoren[1].positie[0]:.1f}, {sys.actoren[1].positie[1]:.1f}, {sys.actoren[1].positie[2]:.1f}]")
    print(f"    alpha AB→DM: {sys.alpha_matrix[1,0]:.3f}")
    print(f"    Eff coherence: {coherence_hist[-1]:.3f}")

    # Did the system recover?
    ab_final_agency = ab_arr[-1, 1]
    ab_final_trust = ab_arr[-1, 2]
    alpha_final = alpha_hist[-1]
    recovered = alpha_final > 0 and ab_final_trust > 5.0

    print(f"\n  HYSTERESIS TEST:")
    print(f"    Structure improved: YES (DM closure reduced, AB agency increased, W symmetrized)")
    print(f"    Identity resolved:  {'YES' if recovered else 'NO'} (alpha final = {alpha_final:.3f})")
    print(f"    System recovered:   {'YES' if recovered else 'NO — identity persists despite structural reform'}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t_reform = n_steps_phase1

    # Trajectory
    ax = axes[0, 0]
    ax.scatter(dm_arr[:t_reform, 0], dm_arr[:t_reform, 1], c='#e74c3c', s=2, alpha=0.4, label='DM (opposition)')
    ax.scatter(ab_arr[:t_reform, 0], ab_arr[:t_reform, 1], c='#2c3e50', s=2, alpha=0.4, label='AB (opposition)')
    ax.scatter(dm_arr[t_reform:, 0], dm_arr[t_reform:, 1], c='#e74c3c', s=2, alpha=0.8, marker='^')
    ax.scatter(ab_arr[t_reform:, 0], ab_arr[t_reform:, 1], c='#2c3e50', s=2, alpha=0.8, marker='^')
    ax.axvline(x=dm_arr[t_reform, 0], color='green', linestyle='--', alpha=0.5, label='Reform starts')
    ax.set_xlabel('Decision Closure')
    ax.set_ylabel('Agency Allocation')
    ax.set_title('Trajectory: Closure x Agency')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Positions over time
    ax = axes[0, 1]
    t = range(len(dm_arr))
    ax.plot(dm_arr[:, 0], color='#e74c3c', linewidth=1.5, label='DM Closure')
    ax.plot(ab_arr[:, 1], color='#2c3e50', linewidth=1.5, label='AB Agency')
    ax.plot(ab_arr[:, 2], color='#3498db', linewidth=1, alpha=0.7, label='AB Trust')
    ax.axvline(x=t_reform, color='green', linestyle='--', alpha=0.7, label='Helena Voss arrives')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position')
    ax.set_title('Dimensions Over Time')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Alpha over time
    ax = axes[1, 0]
    ax.plot(alpha_hist, color='#c0392b', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=t_reform, color='green', linestyle='--', alpha=0.7, label='Helena Voss arrives')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('alpha (AB toward DM)')
    ax.set_title('Relational Coefficient: The Persistence of Identity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Coherence over time
    ax = axes[1, 1]
    ax.plot(coherence_hist, color='#27ae60', linewidth=1.5)
    ax.axvline(x=t_reform, color='green', linestyle='--', alpha=0.7, label='Helena Voss arrives')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Effective Coherence')
    ax.set_title('System Coherence: Does It Recover?')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.suptitle('Helena Voss Effect: Architecture Improves, Identity Persists\n(Hysteresis in Advisory Drift)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'hysteresis_voss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {OUT}/hysteresis_voss.png")

    return {
        'dm_trajectory': dm_arr,
        'ab_trajectory': ab_arr,
        'coherence': coherence_hist,
        'alpha': alpha_hist,
        'recovered': recovered,
    }


# ═══════════════════════════════════════════════════
# EXPERIMENT 3.6: Jonas Reiter — Socialization
# ═══════════════════════════════════════════════════

def experiment_socialization():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3.6: Jonas Reiter — Socialization Effect")
    print("=" * 60)

    d = np.array([-1, 1, 1]) / np.sqrt(3)

    # Established oppositional advisory body with 4 members + DM
    sys = make_advisory_system(
        C_level=8.0, G_level=2.0,
        alpha_ab_dm=-0.06, alpha_dm_ab=-0.04,
        W_dm_to_ab=0.5, W_ab_to_dm=0.05,
        eta=0.03,
        n_extra_members=3,
    )

    # Run to establish oppositional culture
    sys.simuleer(100)

    # Now add Jonas Reiter: neutral alpha, fresh
    reiter = Actor("Reiter", np.array([4.0, 5.0, 7.0]), '#1abc9c')
    sys.actoren.append(reiter)
    n = len(sys.actoren)
    sys.n_actoren = n

    # Expand all matrices
    old_W = sys.W.copy()
    sys.W = np.zeros((n, n))
    sys.W[:n-1, :n-1] = old_W
    # Reiter is influenced by existing members
    sys.W[n-1, 1] = 0.20  # Advisory Body → Reiter
    for i in range(2, n-1):
        sys.W[n-1, i] = 0.15  # Other members → Reiter
    sys.W[n-1, 0] = 0.10  # DM → Reiter (institutional)
    # Reiter's influence on others (initially low)
    sys.W[1, n-1] = 0.05  # Reiter → AB
    sys.W[0, n-1] = 0.02  # Reiter → DM

    old_Wf = sys.W_fixed.copy()
    sys.W_fixed = np.zeros((n, n))
    sys.W_fixed[:n-1, :n-1] = old_Wf[:n-1, :n-1] if old_Wf.shape[0] >= n-1 else old_Wf

    # Reiter's U_config: neutral toward DM, wants to do good work
    sys.U_config["Reiter"] = {
        'doel': np.array([3.0, 7.0, 8.0]),  # Idealistic: wants open decisions, high agency, trust
        'gewicht': 0.35,
        'alpha': {
            'Decision-Maker': 0.0,     # Neutral initially
            'Advisory Body': 0.05,     # Mild positive toward colleagues
        },
    }
    sys._build_alpha_matrix()

    old_alpha = sys.alpha_matrix.copy()

    # Track Reiter's alpha toward DM over time
    reiter_alpha_hist = []
    reiter_agency_hist = []
    reiter_trust_hist = []

    n_socialization_steps = 200

    for t in range(n_socialization_steps):
        reiter_idx = n - 1
        reiter_alpha_hist.append(sys.alpha_matrix[reiter_idx, 0])
        reiter_agency_hist.append(sys.actoren[reiter_idx].positie[1])
        reiter_trust_hist.append(sys.actoren[reiter_idx].positie[2])

        sys.stap()

        # Socialization: Reiter's alpha toward DM shifts based on peer influence
        # This happens through W-evolution (eta=0.03): aligned forces strengthen mutual influence
        # And through the relational dynamics: if peers are in conflict with DM,
        # Reiter experiences the same structural pressures

        # Explicitly model socialization: Reiter adopts average alpha of peers
        if t % 20 == 19:
            # Every 20 steps: peer influence on Reiter's alpha
            peer_alphas = [sys.alpha_matrix[i, 0] for i in range(1, n-1)]
            avg_peer_alpha = np.mean(peer_alphas)
            current_alpha = sys.alpha_matrix[reiter_idx, 0]
            # Move 20% toward peer average
            new_alpha = current_alpha + 0.20 * (avg_peer_alpha - current_alpha)
            sys.U_config["Reiter"]['alpha']['Decision-Maker'] = new_alpha
            sys._build_alpha_matrix()

    print(f"\n  Reiter alpha trajectory:")
    print(f"    t=0:    {reiter_alpha_hist[0]:>+.3f} (neutral)")
    print(f"    t=50:   {reiter_alpha_hist[min(50, len(reiter_alpha_hist)-1)]:>+.3f}")
    print(f"    t=100:  {reiter_alpha_hist[min(100, len(reiter_alpha_hist)-1)]:>+.3f}")
    print(f"    t=200:  {reiter_alpha_hist[-1]:>+.3f}")

    socialized = reiter_alpha_hist[-1] < -0.03
    print(f"\n  Socialization complete: {'YES' if socialized else 'NO'}")
    print(f"  Reiter adopted oppositional posture: {'YES' if socialized else 'NOT YET'}")
    time_to_opposition = next((t for t, a in enumerate(reiter_alpha_hist) if a < -0.03), None)
    if time_to_opposition:
        print(f"  Time to opposition: t={time_to_opposition}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(reiter_alpha_hist, color='#1abc9c', linewidth=2, label='Reiter alpha toward DM')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axhline(y=-0.03, color='red', linestyle='--', alpha=0.5, label='Opposition threshold')
    peer_alpha = sys.alpha_matrix[1, 0]
    ax1.axhline(y=peer_alpha, color='#2c3e50', linestyle=':', alpha=0.5, label=f'Peer average ({peer_alpha:.3f})')
    ax1.set_xlabel('Time Step (after joining)')
    ax1.set_ylabel('alpha (Reiter toward DM)')
    ax1.set_title('Jonas Reiter: Socialization into Oppositional Posture')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    ax2.plot(reiter_agency_hist, color='#2c3e50', linewidth=1.5, label='Agency')
    ax2.plot(reiter_trust_hist, color='#3498db', linewidth=1.5, label='Trust')
    ax2.set_xlabel('Time Step (after joining)')
    ax2.set_ylabel('Position')
    ax2.set_title('Reiter Position Evolution')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.suptitle('Jonas Reiter Effect: New Members Absorb Institutional Identity',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'socialization_reiter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {OUT}/socialization_reiter.png")

    return {
        'alpha_trajectory': reiter_alpha_hist,
        'socialized': socialized,
        'time_to_opposition': time_to_opposition,
    }


# ═══════════════════════════════════════════════════
# RUN ALL EXPERIMENTS
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print("ADVISORY DRIFT — TYPOLOGY & DRIFT SIMULATIONS")
    print("=" * 60)

    r1 = experiment_four_typologies()
    r2 = experiment_drift_mechanism()
    r3 = experiment_aadd_intervention()
    r4 = experiment_hysteresis()
    r5 = experiment_socialization()

    print("\n" + "=" * 60)
    print("ALL ADVISORY DRIFT EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved in: {OUT}/")
    print("  - four_typologies.png")
    print("  - drift_mechanism.png")
    print("  - aadd_intervention.png")
    print("  - hysteresis_voss.png")
    print("  - socialization_reiter.png")
