"""
Advisory Drift — ODE Continuous-Time Formulation
Experiment 4.3: Phase portraits, bifurcation diagrams, hysteresis loops

Three-state ODE system:
  dx/dt = α·C·(1-G)·(1-D)·R - β·G·D·(1-x) + γ·z - δ·y·x     (adversarial posture)
  dy/dt = -ε·x·O·y + ζ·G·(1-x)                                  (trust)
  dz/dt = θ·max(0, x-x_crit)·(1-z) - ι·(1-x)·z                  (identity crystallization)

State variables:
  x ∈ [0,1]: adversarial posture (0 = cooperative, 1 = fully oppositional)
  y ∈ [0,1]: trust in decision-maker (0 = none, 1 = full)
  z ∈ [0,1]: identity crystallization (0 = fluid, 1 = locked)

Structural parameters (from Advisory Drift Ch 6-8):
  C: Decision Closure (0-1)
  O: Mandatory Leverage (0-1)
  G: Agency Allocation (0-1)
  D: Domain Clarity (0-1)
  R: Risk Asymmetry (0-1)

Dynamic parameters:
  α: closure-driven opposition rate
  β: agency-driven cooperation rate
  γ: identity reinforcement of opposition
  δ: trust-mediated opposition dampening
  ε: opposition-driven trust erosion
  ζ: agency-driven trust building
  θ: identity crystallization rate
  ι: identity dissolution rate
  x_crit: opposition threshold for identity formation
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

# ── Default ODE parameters ──

DEFAULT_PARAMS = {
    'alpha': 0.8,    # closure-driven opposition rate
    'beta': 0.6,     # agency-driven cooperation rate
    'gamma': 0.5,    # identity reinforcement of opposition
    'delta': 0.4,    # trust-mediated opposition dampening
    'epsilon': 0.7,  # opposition-driven trust erosion
    'zeta': 0.3,     # agency-driven trust building
    'theta': 0.4,    # identity crystallization rate
    'iota': 0.1,     # identity dissolution rate
    'x_crit': 0.4,   # opposition threshold for identity formation
}


def advisory_drift_ode(t, state, C, O, G, D, R, params=None):
    """Advisory Drift ODE system.

    Returns dx/dt, dy/dt, dz/dt.
    """
    x, y, z = state
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    z = np.clip(z, 0, 1)

    p = params or DEFAULT_PARAMS
    a, b, g, d = p['alpha'], p['beta'], p['gamma'], p['delta']
    e, ze, th, io = p['epsilon'], p['zeta'], p['theta'], p['iota']
    x_c = p['x_crit']

    # dx/dt: adversarial posture
    # Increases with: closure × lack of agency × lack of clarity × risk, identity reinforcement
    # Decreases with: agency × clarity × cooperation pull, trust dampening
    dxdt = a * C * (1 - G) * (1 - D) * R - b * G * D * (1 - x) + g * z - d * y * x

    # dy/dt: trust
    # Decreases with: opposition × mandatory leverage (forced engagement erodes trust)
    # Increases with: agency × cooperation (genuine engagement builds trust)
    dydt = -e * x * O * y + ze * G * (1 - x)

    # dz/dt: identity crystallization
    # Increases when opposition exceeds threshold (sustained opposition crystallizes)
    # Decreases slowly when opposition is low (but very slowly — hysteresis)
    dzdt = th * max(0, x - x_c) * (1 - z) - io * max(0, (1 - x)) * z

    return [dxdt, dydt, dzdt]


def simulate_ode(C, O, G, D, R, x0=0.1, y0=0.8, z0=0.0,
                 t_span=(0, 50), n_points=1000, params=None):
    """Simulate the ODE and return solution."""
    sol = solve_ivp(
        advisory_drift_ode,
        t_span,
        [x0, y0, z0],
        args=(C, O, G, D, R, params),
        t_eval=np.linspace(t_span[0], t_span[1], n_points),
        method='RK45',
        max_step=0.1,
    )
    # Clip to valid range
    sol.y = np.clip(sol.y, 0, 1)
    return sol


def find_equilibria(C, O, G, D, R, params=None, n_guesses=50):
    """Find equilibrium points by trying multiple initial guesses."""
    equilibria = []

    for _ in range(n_guesses):
        guess = np.random.uniform(0, 1, 3)
        try:
            eq, info, ier, msg = fsolve(
                lambda s: advisory_drift_ode(0, s, C, O, G, D, R, params),
                guess, full_output=True
            )
            if ier == 1:  # converged
                eq = np.clip(eq, 0, 1)
                # Check it's actually an equilibrium
                residual = np.linalg.norm(advisory_drift_ode(0, eq, C, O, G, D, R, params))
                if residual < 1e-6:
                    # Check if new
                    is_new = True
                    for existing in equilibria:
                        if np.allclose(eq, existing, atol=0.05):
                            is_new = False
                            break
                    if is_new:
                        equilibria.append(eq)
        except Exception:
            pass

    return equilibria


def classify_equilibrium(eq, C, O, G, D, R, params=None, eps=1e-5):
    """Classify equilibrium stability via Jacobian eigenvalues."""
    x, y, z = eq
    J = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            state_plus = list(eq)
            state_minus = list(eq)
            state_plus[j] += eps
            state_minus[j] -= eps
            f_plus = advisory_drift_ode(0, state_plus, C, O, G, D, R, params)
            f_minus = advisory_drift_ode(0, state_minus, C, O, G, D, R, params)
            J[i, j] = (f_plus[i] - f_minus[i]) / (2 * eps)

    eigenvalues = np.linalg.eigvals(J)
    real_parts = eigenvalues.real

    if all(r < -1e-8 for r in real_parts):
        return 'stable', eigenvalues
    elif all(r > 1e-8 for r in real_parts):
        return 'unstable', eigenvalues
    else:
        return 'saddle', eigenvalues


# ═══════════════════════════════════════════════════
# EXPERIMENT 4.3a: Phase Portraits per Typology
# ═══════════════════════════════════════════════════

def experiment_phase_portraits():
    print("=" * 60)
    print("EXPERIMENT 4.3a: Phase Portraits per Typology")
    print("=" * 60)

    typologies = {
        'Type I: Open Co-Design': {
            'C': 0.2, 'O': 0.3, 'G': 0.8, 'D': 0.8, 'R': 0.3,
        },
        'Type II: Guided Design': {
            'C': 0.7, 'O': 0.5, 'G': 0.7, 'D': 0.8, 'R': 0.4,
        },
        'Type III: Corrective Open': {
            'C': 0.2, 'O': 0.6, 'G': 0.2, 'D': 0.3, 'R': 0.5,
        },
        'Type IV: Mandatory Corrective': {
            'C': 0.9, 'O': 0.9, 'G': 0.1, 'D': 0.2, 'R': 0.8,
        },
    }

    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    results = {}

    for idx, (name, cfg) in enumerate(typologies.items()):
        C, O, G, D, R = cfg['C'], cfg['O'], cfg['G'], cfg['D'], cfg['R']

        # Find equilibria
        eqs = find_equilibria(C, O, G, D, R)
        eq_info = []
        for eq in eqs:
            stability, eigvals = classify_equilibrium(eq, C, O, G, D, R)
            eq_info.append({'pos': eq, 'stability': stability, 'eigvals': eigvals})

        # Simulate from multiple initial conditions
        trajectories = []
        initial_conditions = [
            (0.1, 0.8, 0.0),  # Cooperative start
            (0.5, 0.5, 0.0),  # Neutral
            (0.8, 0.2, 0.5),  # Already oppositional
            (0.1, 0.9, 0.0),  # High trust
            (0.9, 0.1, 0.8),  # Deep opposition with identity
        ]

        for x0, y0, z0 in initial_conditions:
            sol = simulate_ode(C, O, G, D, R, x0, y0, z0, t_span=(0, 80))
            trajectories.append(sol)

        results[name] = {
            'equilibria': eq_info,
            'trajectories': trajectories,
            'config': cfg,
        }

        # Phase portrait: x vs y (opposition vs trust)
        ax = fig.add_subplot(gs[idx, 0])
        for sol in trajectories:
            ax.plot(sol.y[0], sol.y[1], '-', alpha=0.7, linewidth=1.2)
            ax.plot(sol.y[0, 0], sol.y[1, 0], 'o', markersize=5, color='green')
            ax.plot(sol.y[0, -1], sol.y[1, -1], 's', markersize=7, color='red')
        for ei in eq_info:
            color = '#27ae60' if ei['stability'] == 'stable' else '#e74c3c' if ei['stability'] == 'unstable' else '#f39c12'
            marker = 'o' if ei['stability'] == 'stable' else 'x'
            ax.plot(ei['pos'][0], ei['pos'][1], marker, color=color, markersize=12,
                    markeredgewidth=2, zorder=10)
        ax.set_xlabel('Opposition (x)')
        ax.set_ylabel('Trust (y)')
        ax.set_title(f'{name}\nx vs y', fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

        # Phase portrait: x vs z (opposition vs identity)
        ax = fig.add_subplot(gs[idx, 1])
        for sol in trajectories:
            ax.plot(sol.y[0], sol.y[2], '-', alpha=0.7, linewidth=1.2)
            ax.plot(sol.y[0, 0], sol.y[2, 0], 'o', markersize=5, color='green')
            ax.plot(sol.y[0, -1], sol.y[2, -1], 's', markersize=7, color='red')
        for ei in eq_info:
            color = '#27ae60' if ei['stability'] == 'stable' else '#e74c3c' if ei['stability'] == 'unstable' else '#f39c12'
            marker = 'o' if ei['stability'] == 'stable' else 'x'
            ax.plot(ei['pos'][0], ei['pos'][2], marker, color=color, markersize=12,
                    markeredgewidth=2, zorder=10)
        ax.set_xlabel('Opposition (x)')
        ax.set_ylabel('Identity (z)')
        ax.set_title(f'x vs z', fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

        # Time series
        ax = fig.add_subplot(gs[idx, 2])
        # Use the "cooperative start" trajectory
        sol = trajectories[0]
        ax.plot(sol.t, sol.y[0], '-', color='#e74c3c', linewidth=1.5, label='Opposition (x)')
        ax.plot(sol.t, sol.y[1], '-', color='#3498db', linewidth=1.5, label='Trust (y)')
        ax.plot(sol.t, sol.y[2], '-', color='#8e44ad', linewidth=1.5, label='Identity (z)')
        ax.set_xlabel('Time')
        ax.set_ylabel('State')
        ax.set_title(f'Time series (cooperative start)', fontsize=9)
        ax.set_xlim(0, 80)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='center right')
        ax.grid(True, alpha=0.2)

        # Print equilibria
        n_stable = sum(1 for e in eq_info if e['stability'] == 'stable')
        n_saddle = sum(1 for e in eq_info if e['stability'] == 'saddle')
        print(f"\n  {name}:")
        print(f"    Equilibria found: {len(eqs)} ({n_stable} stable, {n_saddle} saddle)")
        for i, ei in enumerate(eq_info):
            x, y, z = ei['pos']
            print(f"    [{x:.2f}, {y:.2f}, {z:.2f}] — {ei['stability']}")

    plt.suptitle('Advisory Drift ODE: Phase Portraits by Typology\n'
                 'Green circles = stable equilibria, Red X = unstable, Orange = saddle',
                 fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(OUT, 'ode_phase_portraits.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {OUT}/ode_phase_portraits.png")
    return results


# ═══════════════════════════════════════════════════
# EXPERIMENT 4.3b: Bifurcation Diagram — C vs x*
# ═══════════════════════════════════════════════════

def experiment_bifurcation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4.3b: Bifurcation Diagram — C vs x* (simulation-based)")
    print("=" * 60)

    # The system has boundary attractors (x≈0 cooperative, x≈1 oppositional).
    # Instead of finding interior equilibria, simulate from cooperative start
    # and map which basin the system falls into for each (C, G) pair.

    C_values = np.linspace(0.01, 0.99, 50)
    G_levels = [0.1, 0.3, 0.5, 0.7]
    O, D, R = 0.7, 0.3, 0.6

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: final opposition x* vs C for different G (from cooperative start)
    ax = axes[0]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']

    for g_idx, G in enumerate(G_levels):
        final_x = []
        for C in C_values:
            sol = simulate_ode(C, O, G, D, R, x0=0.1, y0=0.8, z0=0.0,
                               t_span=(0, 100), n_points=500)
            final_x.append(sol.y[0, -1])

        ax.plot(C_values, final_x, 'o-', color=colors[g_idx], markersize=3,
                linewidth=1.5, label=f'G={G}')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Decision Closure (C)', fontsize=11)
    ax.set_ylabel('Final Opposition (x*)', fontsize=11)
    ax.set_title('Drift Outcome vs Closure\n(from cooperative start, different Agency levels)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)

    # Right: C-G phase diagram (heatmap of final opposition)
    ax = axes[1]
    C_grid = np.linspace(0.01, 0.99, 40)
    G_grid = np.linspace(0.01, 0.99, 40)
    X_final = np.zeros((len(G_grid), len(C_grid)))

    for i, G in enumerate(G_grid):
        for j, C in enumerate(C_grid):
            sol = simulate_ode(C, O, G, D, R, x0=0.1, y0=0.8, z0=0.0,
                               t_span=(0, 100), n_points=300)
            X_final[i, j] = sol.y[0, -1]

    im = ax.imshow(X_final, extent=[0, 1, 0, 1], origin='lower', aspect='auto',
                   cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Final Opposition (x*)')

    # Draw contour at x=0.5 (boundary between cooperation and opposition)
    contour = ax.contour(C_grid, G_grid, X_final, levels=[0.5],
                         colors='white', linewidths=2, linestyles='--')
    ax.clabel(contour, fmt='x*=0.5', fontsize=8)

    # Mark the four typologies
    type_markers = {
        'I': (0.2, 0.8), 'II': (0.7, 0.7),
        'III': (0.2, 0.2), 'IV': (0.9, 0.1),
    }
    for label, (c, g) in type_markers.items():
        ax.plot(c, g, 'ws', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        ax.text(c, g + 0.05, label, ha='center', fontsize=9, fontweight='bold',
                color='white', path_effects=[
                    matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])

    ax.set_xlabel('Decision Closure (C)', fontsize=11)
    ax.set_ylabel('Agency Allocation (G)', fontsize=11)
    ax.set_title('C-G Phase Diagram\n(Red = opposition, Green = cooperation)')
    ax.grid(True, alpha=0.15, color='white')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'ode_bifurcation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Find critical C for each G
    print(f"\n  Critical C (threshold for opposition) by G:")
    for g_idx, G in enumerate(G_levels):
        for j, C in enumerate(C_values):
            sol = simulate_ode(C, O, G, D, R, x0=0.1, y0=0.8, z0=0.0,
                               t_span=(0, 100), n_points=300)
            if sol.y[0, -1] > 0.5:
                print(f"    G={G}: C* ~= {C:.2f}")
                break
        else:
            print(f"    G={G}: C* > 0.99 (no drift)")

    print(f"\nFigure saved: {OUT}/ode_bifurcation.png")
    return {'X_final': X_final, 'C_grid': C_grid, 'G_grid': G_grid}


# ═══════════════════════════════════════════════════
# EXPERIMENT 4.3c: Hysteresis Loop
# ═══════════════════════════════════════════════════

def experiment_hysteresis_loop():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4.3c: Hysteresis Loop — C Increase then Decrease")
    print("=" * 60)

    O, G, D, R = 0.7, 0.2, 0.3, 0.6
    dt = 0.1
    n_steps = 800

    # Phase 1: C increases from 0.1 to 0.9
    # Phase 2: C decreases from 0.9 to 0.1
    C_trajectory = np.concatenate([
        np.linspace(0.1, 0.9, n_steps // 2),
        np.linspace(0.9, 0.1, n_steps // 2),
    ])

    # Simulate the ODE with slowly changing C
    x, y, z = 0.1, 0.8, 0.0  # Start cooperative
    x_hist, y_hist, z_hist = [x], [y], [z]
    C_hist = [C_trajectory[0]]

    for i in range(1, len(C_trajectory)):
        C = C_trajectory[i]
        # Take a few integration steps at this C
        for _ in range(5):
            dxdt, dydt, dzdt = advisory_drift_ode(0, [x, y, z], C, O, G, D, R)
            x = np.clip(x + dt * dxdt, 0, 1)
            y = np.clip(y + dt * dydt, 0, 1)
            z = np.clip(z + dt * dzdt, 0, 1)

        x_hist.append(x)
        y_hist.append(y)
        z_hist.append(z)
        C_hist.append(C)

    x_hist = np.array(x_hist)
    y_hist = np.array(y_hist)
    z_hist = np.array(z_hist)
    C_hist = np.array(C_hist)

    half = n_steps // 2

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Hysteresis loop: C vs x
    ax = axes[0, 0]
    ax.plot(C_hist[:half+1], x_hist[:half+1], '-', color='#e74c3c', linewidth=2,
            label='C increasing (drift)')
    ax.plot(C_hist[half:], x_hist[half:], '--', color='#3498db', linewidth=2,
            label='C decreasing (reform)')
    ax.annotate('Start', xy=(C_hist[0], x_hist[0]), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='green'),
                xytext=(C_hist[0]+0.1, x_hist[0]+0.1))
    ax.annotate('Peak\nclosure', xy=(C_hist[half], x_hist[half]), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray'),
                xytext=(C_hist[half]-0.2, x_hist[half]-0.15))
    ax.set_xlabel('Decision Closure (C)', fontsize=11)
    ax.set_ylabel('Opposition (x)', fontsize=11)
    ax.set_title('Hysteresis Loop: Opposition vs Closure')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Hysteresis loop: C vs z (identity)
    ax = axes[0, 1]
    ax.plot(C_hist[:half+1], z_hist[:half+1], '-', color='#8e44ad', linewidth=2,
            label='C increasing')
    ax.plot(C_hist[half:], z_hist[half:], '--', color='#1abc9c', linewidth=2,
            label='C decreasing')
    ax.set_xlabel('Decision Closure (C)', fontsize=11)
    ax.set_ylabel('Identity Crystallization (z)', fontsize=11)
    ax.set_title('Hysteresis: Identity vs Closure')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Time series of all states
    ax = axes[1, 0]
    t = range(len(x_hist))
    ax.plot(t, x_hist, '-', color='#e74c3c', linewidth=1.5, label='Opposition (x)')
    ax.plot(t, y_hist, '-', color='#3498db', linewidth=1.5, label='Trust (y)')
    ax.plot(t, z_hist, '-', color='#8e44ad', linewidth=1.5, label='Identity (z)')
    ax.axvline(x=half, color='green', linestyle='--', alpha=0.7, label='Reform begins')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('State', fontsize=11)
    ax.set_title('State Evolution: Drift then Reform')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # C trajectory
    ax = axes[1, 1]
    ax.plot(t, C_hist, '-', color='#2c3e50', linewidth=2)
    ax.axvline(x=half, color='green', linestyle='--', alpha=0.7, label='Reform begins')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Decision Closure (C)', fontsize=11)
    ax.set_title('Closure Trajectory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.suptitle('Advisory Drift Hysteresis: Closure Increases then Decreases\n'
                 'Opposition and identity persist after structural reform — the path matters',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'ode_hysteresis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Quantify hysteresis
    # At C=0.5 on the way up vs way down
    idx_up = np.argmin(np.abs(C_hist[:half] - 0.5))
    idx_down = half + np.argmin(np.abs(C_hist[half:] - 0.5))
    hysteresis_x = x_hist[idx_down] - x_hist[idx_up]
    hysteresis_z = z_hist[idx_down] - z_hist[idx_up]

    print(f"\n  Hysteresis at C=0.5:")
    print(f"    Opposition (x): up={x_hist[idx_up]:.3f}, down={x_hist[idx_down]:.3f}, gap={hysteresis_x:.3f}")
    print(f"    Identity (z):   up={z_hist[idx_up]:.3f}, down={z_hist[idx_down]:.3f}, gap={hysteresis_z:.3f}")
    print(f"    Trust (y):      up={y_hist[idx_up]:.3f}, down={y_hist[idx_down]:.3f}")
    print(f"\n  Final state (C returned to 0.1):")
    print(f"    Opposition: {x_hist[-1]:.3f} (started at {x_hist[0]:.3f})")
    print(f"    Trust:      {y_hist[-1]:.3f} (started at {y_hist[0]:.3f})")
    print(f"    Identity:   {z_hist[-1]:.3f} (started at {z_hist[0]:.3f})")
    recovered = x_hist[-1] < 0.2 and y_hist[-1] > 0.5
    print(f"\n  System recovered: {'YES' if recovered else 'NO — hysteresis confirmed'}")
    print(f"\nFigure saved: {OUT}/ode_hysteresis.png")

    return {
        'x': x_hist, 'y': y_hist, 'z': z_hist, 'C': C_hist,
        'hysteresis_x': hysteresis_x, 'hysteresis_z': hysteresis_z,
    }


# ═══════════════════════════════════════════════════
# EXPERIMENT 4.3d: Sensitivity of Critical C to Parameters
# ═══════════════════════════════════════════════════

def experiment_parameter_sensitivity():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4.3d: Critical C Sensitivity to COGDR")
    print("=" * 60)

    # For each AD variable, sweep it and find critical C
    base = {'C': 0.5, 'O': 0.7, 'G': 0.3, 'D': 0.3, 'R': 0.6}

    variables = {
        'O (Mandatory Leverage)': ('O', np.linspace(0.1, 0.9, 30)),
        'G (Agency Allocation)': ('G', np.linspace(0.05, 0.95, 30)),
        'D (Domain Clarity)': ('D', np.linspace(0.05, 0.95, 30)),
        'R (Risk Asymmetry)': ('R', np.linspace(0.1, 0.9, 30)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#e74c3c', '#27ae60', '#3498db', '#f39c12']

    for idx, (label, (var_key, var_range)) in enumerate(variables.items()):
        ax = axes[idx // 2, idx % 2]

        final_x_values = []

        for val in var_range:
            cfg = base.copy()
            cfg[var_key] = val

            # Simulate from cooperative start
            sol = simulate_ode(cfg['C'], cfg['O'], cfg['G'], cfg['D'], cfg['R'],
                               x0=0.1, y0=0.8, z0=0.0, t_span=(0, 100))
            final_x = sol.y[0, -1]
            final_y = sol.y[1, -1]
            final_z = sol.y[2, -1]

            final_x_values.append(final_x)

        ax.plot(var_range, final_x_values, 'o-', color=colors[idx], markersize=4, linewidth=1.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Opposition threshold')
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Final Opposition (x*)', fontsize=10)
        ax.set_title(f'Effect of {label} on Drift')
        ax.set_xlim(var_range[0], var_range[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Advisory Drift ODE: Sensitivity of Opposition to Each Structural Variable\n'
                 '(All other parameters at baseline: C=0.5, O=0.7, G=0.3, D=0.3, R=0.6)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'ode_parameter_sensitivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {OUT}/ode_parameter_sensitivity.png")

    return final_x_values


# ═══════════════════════════════════════════════════
# EXPERIMENT 4.3e: Complete Typology Comparison
# ═══════════════════════════════════════════════════

def experiment_typology_comparison():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4.3e: Typology Time Series Comparison")
    print("=" * 60)

    typologies = {
        'Type I: Open Co-Design': {'C': 0.2, 'O': 0.3, 'G': 0.8, 'D': 0.8, 'R': 0.3},
        'Type II: Guided Design': {'C': 0.7, 'O': 0.5, 'G': 0.7, 'D': 0.8, 'R': 0.4},
        'Type III: Corrective Open': {'C': 0.2, 'O': 0.6, 'G': 0.2, 'D': 0.3, 'R': 0.5},
        'Type IV: Mandatory Corrective': {'C': 0.9, 'O': 0.9, 'G': 0.1, 'D': 0.2, 'R': 0.8},
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    type_colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']

    results = {}

    for idx, (name, cfg) in enumerate(typologies.items()):
        ax = axes[idx // 2, idx % 2]

        # Simulate from cooperative start
        sol = simulate_ode(cfg['C'], cfg['O'], cfg['G'], cfg['D'], cfg['R'],
                           x0=0.1, y0=0.8, z0=0.0, t_span=(0, 80))

        ax.plot(sol.t, sol.y[0], '-', color='#e74c3c', linewidth=2, label='Opposition (x)')
        ax.plot(sol.t, sol.y[1], '-', color='#3498db', linewidth=2, label='Trust (y)')
        ax.plot(sol.t, sol.y[2], '-', color='#8e44ad', linewidth=2, label='Identity (z)')

        final_x, final_y, final_z = sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]
        outcome = "COOPERATIVE" if final_x < 0.3 else "OPPOSITIONAL" if final_x > 0.6 else "MIXED"

        ax.set_xlabel('Time')
        ax.set_ylabel('State')
        ax.set_title(f'{name}\n[C={cfg["C"]}, G={cfg["G"]}, D={cfg["D"]}] → {outcome}',
                     fontsize=9)
        ax.set_xlim(0, 80)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='center right')
        ax.grid(True, alpha=0.2)

        results[name] = {
            'final_x': final_x, 'final_y': final_y, 'final_z': final_z,
            'outcome': outcome,
        }

        print(f"  {name}:")
        print(f"    Final: x={final_x:.3f}, y={final_y:.3f}, z={final_z:.3f} → {outcome}")

    plt.suptitle('Advisory Drift ODE: Four Typologies from Cooperative Start\n'
                 'Same initial conditions (x=0.1, y=0.8, z=0.0), different structural parameters',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'ode_typology_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {OUT}/ode_typology_comparison.png")
    return results


# ═══════════════════════════════════════════════════
# RUN ALL EXPERIMENTS
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print("ADVISORY DRIFT — ODE CONTINUOUS-TIME DYNAMICS")
    print("=" * 60)

    r1 = experiment_phase_portraits()
    r2 = experiment_bifurcation()
    r3 = experiment_hysteresis_loop()
    r4 = experiment_parameter_sensitivity()
    r5 = experiment_typology_comparison()

    print("\n" + "=" * 60)
    print("ALL ODE EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved in: {OUT}/")
    print("  - ode_phase_portraits.png")
    print("  - ode_bifurcation.png")
    print("  - ode_hysteresis.png")
    print("  - ode_parameter_sensitivity.png")
    print("  - ode_typology_comparison.png")
