"""
SSRN Paper — Sensitivity Analysis & Robustness Experiments
Must-haves: W-perturbation, α-sweep, multi-expert
Nice-to-haves: γ-sensitivity, dimension sensitivity
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beleidsdynamica_v4 import Krachtenveld, Actor
from beleidsdynamica_v4.analyse import diagnose

np.random.seed(42)
OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

# ── Passend Onderwijs baseline configuration (from v4 rapport) ──

def make_passend_onderwijs(alpha_leraar_min=-0.05, alpha_leraar_swv=-0.03,
                            alpha_leraar_school=-0.02, W_scale=1.0,
                            gamma_leraar=0.4, eta=0.01):
    """Create Passend Onderwijs configuration. Returns Krachtenveld.

    Parameters
    ----------
    alpha_leraar_min : float
        Teacher→Ministry relational coefficient (negative = conflict)
    alpha_leraar_swv : float
        Teacher→SWV relational coefficient
    alpha_leraar_school : float
        Teacher→School relational coefficient
    W_scale : float
        Multiplier on the W matrix
    gamma_leraar : float
        Teacher's goal weight (intrinsic motivation strength)
    eta : float
        W-evolution learning rate
    """
    actoren = [
        Actor("Leraar", np.array([4.0, 6.5, 5.5]), '#27ae60'),
        Actor("School", np.array([4.5, 5.5, 5.0]), '#f39c12'),
        Actor("SWV", np.array([4.0, 5.0, 5.5]), '#9b59b6'),
        Actor("Ministerie", np.array([3.5, 6.0, 5.0]), '#e74c3c'),
    ]

    U_config = {
        "Leraar": {
            'doel': np.array([3.0, 7.5, 7.0]),
            'gewicht': gamma_leraar,
            'alpha': {
                'School': alpha_leraar_school,
                'SWV': alpha_leraar_swv,
                'Ministerie': alpha_leraar_min,
            }
        },
        "School": {
            'doel': np.array([5.0, 5.0, 6.0]),
            'gewicht': 0.4,
            'alpha': {
                'Leraar': 0.05,
                'SWV': 0.02,
                'Ministerie': 0.05,
            }
        },
        "SWV": {
            'doel': np.array([5.0, 6.0, 5.5]),
            'gewicht': 0.3,
            'alpha': {
                'Leraar': 0.0,
                'School': 0.05,
                'Ministerie': 0.08,
            }
        },
        "Ministerie": {
            'doel': np.array([6.0, 4.0, 4.0]),
            'gewicht': 0.5,
            'alpha': {
                'Leraar': 0.0,
                'School': 0.02,
                'SWV': 0.05,
            }
        },
    }

    W = np.array([
        [0.0, 0.5, 0.2, 0.3],   # Leraar: heavily influenced
        [0.1, 0.0, 0.3, 0.4],   # School: influenced by higher levels
        [0.0, 0.2, 0.0, 0.4],   # SWV: influenced by ministry
        [0.0, 0.1, 0.1, 0.0],   # Ministerie: minimal feedback
    ]) * W_scale

    C = {
        "Leraar": {0: (0, 10), 1: (0, 10), 2: (0, 10)},
        "School": {0: (0, 10), 1: (0, 10), 2: (0, 10)},
        "SWV": {0: (0, 10), 1: (0, 10), 2: (0, 10)},
        "Ministerie": {0: (0, 10), 1: (0, 10), 2: (0, 10)},
    }

    return Krachtenveld(actoren, U_config, W, C=C, eta=eta)


def run_attractor_analysis(system, n_starts=15):
    """Run attractor analysis, return count and positions."""
    attractors = system.vind_attractoren_multi(n_starts=n_starts)
    return len(attractors), attractors


def get_diagnostics(system, d=None, test_attractoren=True):
    """Get full diagnostics."""
    if d is None:
        d = np.array([-1, 1, 1]) / np.sqrt(3)
    diag = diagnose(system, d_gewenst=d, test_attractoren=test_attractoren)
    return {
        'asymmetrie_W': diag.asymmetrie_W,
        'coherentie': diag.coherentie,
        'eff_coherentie': diag.effectieve_coherentie,
        'n_attractoren': diag.n_attractoren,
        'samenwerking': diag.samenwerking_score,
    }


# ═══════════════════════════════════════════════════
# EXPERIMENT 1.1: W-Matrix Perturbation
# ═══════════════════════════════════════════════════

def experiment_W_perturbation():
    print("=" * 60)
    print("EXPERIMENT 1.1: W-Matrix Perturbation")
    print("=" * 60)

    perturbation_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    n_trials = 10
    results = {p: [] for p in perturbation_levels}

    baseline = make_passend_onderwijs()
    W_base = baseline.W.copy()

    for p in perturbation_levels:
        for trial in range(n_trials):
            sys_copy = make_passend_onderwijs()
            # Perturb W
            noise = np.random.uniform(-p, p, W_base.shape)
            np.fill_diagonal(noise, 0)
            sys_copy.W = np.clip(W_base + noise, 0, 2.0)

            n_att, _ = run_attractor_analysis(sys_copy, n_starts=12)
            diag = get_diagnostics(sys_copy, test_attractoren=False)
            results[p].append({
                'n_attractoren': n_att,
                'eff_coherentie': diag['eff_coherentie'],
                'asymmetrie': diag['asymmetrie_W'],
            })

    # Summarize
    print(f"\n{'Perturbation':>12} | {'Attractors (mean+/-std)':>24} | {'Eff.Coh (mean)':>14} | {'Robust?':>8}")
    print("-" * 70)
    baseline_att = np.mean([r['n_attractoren'] for r in results[0.0]])
    for p in perturbation_levels:
        atts = [r['n_attractoren'] for r in results[p]]
        ecoh = [r['eff_coherentie'] for r in results[p]]
        robust = "YES" if abs(np.mean(atts) - baseline_att) < 1.5 else "NO"
        print(f"{p:>12.2f} | {np.mean(atts):>8.1f} +/- {np.std(atts):>4.1f}          | {np.mean(ecoh):>14.3f} | {robust:>8}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    means = [np.mean([r['n_attractoren'] for r in results[p]]) for p in perturbation_levels]
    stds = [np.std([r['n_attractoren'] for r in results[p]]) for p in perturbation_levels]
    ax1.errorbar(perturbation_levels, means, yerr=stds, marker='o', capsize=4, color='#2c3e50')
    ax1.axhline(y=baseline_att, color='red', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_att:.0f})')
    ax1.set_xlabel('W Perturbation Magnitude')
    ax1.set_ylabel('Number of Attractors')
    ax1.set_title('Attractor Count Robustness to W Perturbation')
    ax1.legend()

    ecoh_means = [np.mean([r['eff_coherentie'] for r in results[p]]) for p in perturbation_levels]
    ecoh_stds = [np.std([r['eff_coherentie'] for r in results[p]]) for p in perturbation_levels]
    ax2.errorbar(perturbation_levels, ecoh_means, yerr=ecoh_stds, marker='s', capsize=4, color='#e74c3c')
    ax2.set_xlabel('W Perturbation Magnitude')
    ax2.set_ylabel('Effective Coherence')
    ax2.set_title('Effective Coherence Robustness')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'W_perturbation.png'), dpi=150)
    plt.close()
    print(f"\nFigure saved: {OUT}/W_perturbation.png")
    return results


# ═══════════════════════════════════════════════════
# EXPERIMENT 1.2: alpha-Parameter Sweep (Proposition 3)
# ═══════════════════════════════════════════════════

def experiment_alpha_sweep():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1.2: alpha-Parameter Sweep (Proposition 3)")
    print("=" * 60)

    alpha_values = np.arange(-0.10, 0.11, 0.01)
    results = []

    for a in alpha_values:
        sys = make_passend_onderwijs(alpha_leraar_min=a,
                                      alpha_leraar_swv=a * 0.6,
                                      alpha_leraar_school=a * 0.4)
        n_att, _ = run_attractor_analysis(sys, n_starts=12)
        diag = get_diagnostics(sys, test_attractoren=False)

        # Also simulate to get final positions
        sys2 = make_passend_onderwijs(alpha_leraar_min=a,
                                       alpha_leraar_swv=a * 0.6,
                                       alpha_leraar_school=a * 0.4)
        sys2.simuleer(200)

        leraar_final = sys2.actoren[0].positie.copy()  # Leraar is index 0
        results.append({
            'alpha': a,
            'n_attractoren': n_att,
            'eff_coherentie': diag['eff_coherentie'],
            'samenwerking': diag['samenwerking'],
            'leraar_autonomie': leraar_final[1],
            'leraar_werkdruk': leraar_final[0],
        })

    # Find bifurcation point
    print(f"\n{'alpha':>6} | {'Attractors':>10} | {'Eff.Coh':>8} | {'Teacher Auto.':>13} | {'Phase':>10}")
    print("-" * 60)
    for r in results:
        phase = "CONFLICT" if r['alpha'] < -0.03 else ("NEUTRAL" if r['alpha'] < 0.03 else "COOPERATE")
        print(f"{r['alpha']:>6.2f} | {r['n_attractoren']:>10} | {r['eff_coherentie']:>8.3f} | {r['leraar_autonomie']:>13.1f} | {phase:>10}")

    # Find bifurcation
    alphas = [r['alpha'] for r in results]
    n_atts = [r['n_attractoren'] for r in results]
    bifurcation = None
    for i in range(len(n_atts) - 1):
        if n_atts[i] > 2 and n_atts[i+1] <= 2:
            bifurcation = alphas[i]
            break
        if n_atts[i] <= 2 and n_atts[i+1] > 2:
            bifurcation = alphas[i+1]

    if bifurcation:
        print(f"\nBifurcation point: alpha ~= {bifurcation:.2f}")
    print("Proposition 3 test: negative alpha -> multiple attractors?",
          "CONFIRMED" if any(r['n_attractoren'] > 2 for r in results if r['alpha'] < -0.03) else "NOT CONFIRMED")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(alphas, n_atts, 'o-', color='#2c3e50', markersize=4)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    if bifurcation:
        ax.axvline(x=bifurcation, color='red', linestyle='--', alpha=0.7, label=f'Bifurcation ~= {bifurcation:.2f}')
    ax.set_xlabel('alpha (relational coefficient)')
    ax.set_ylabel('Number of Attractors')
    ax.set_title('Attractor Count vs alpha')
    ax.legend()

    ax = axes[1]
    ax.plot(alphas, [r['eff_coherentie'] for r in results], 's-', color='#e74c3c', markersize=4)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('alpha (relational coefficient)')
    ax.set_ylabel('Effective Coherence')
    ax.set_title('Effective Coherence vs alpha')

    ax = axes[2]
    ax.plot(alphas, [r['leraar_autonomie'] for r in results], '^-', color='#27ae60', markersize=4, label='Autonomy')
    ax.plot(alphas, [r['leraar_werkdruk'] for r in results], 'v-', color='#e67e22', markersize=4, label='Workload')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('alpha (relational coefficient)')
    ax.set_ylabel('Teacher Final Position')
    ax.set_title('Teacher Outcomes vs alpha')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'alpha_sweep.png'), dpi=150)
    plt.close()
    print(f"Figure saved: {OUT}/alpha_sweep.png")
    return results


# ═══════════════════════════════════════════════════
# EXPERIMENT 1.3: gamma-Sensitivity (Proposition 1)
# ═══════════════════════════════════════════════════

def experiment_gamma_sensitivity():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1.3: gamma-Sensitivity (Proposition 1)")
    print("=" * 60)

    gamma_values = np.arange(0.05, 0.85, 0.05)
    results_opposed = []

    leraar_doel = np.array([3.0, 7.5, 7.0])

    for g in gamma_values:
        sys = make_passend_onderwijs(gamma_leraar=g)
        sys.simuleer(200)
        diag = get_diagnostics(sys, test_attractoren=False)
        leraar = sys.actoren[0].positie.copy()
        results_opposed.append({
            'gamma': g,
            'eff_coherentie': diag['eff_coherentie'],
            'leraar_autonomie': leraar[1],
            'leraar_strain': np.linalg.norm(leraar - leraar_doel),
        })

    print(f"\n{'gamma':>6} | {'Eff.Coh':>18} | {'Teacher Auto.':>13} | {'Strain':>8}")
    print("-" * 55)
    for r in results_opposed:
        print(f"{r['gamma']:>6.2f} | {r['eff_coherentie']:>18.3f} | {r['leraar_autonomie']:>13.1f} | {r['leraar_strain']:>8.2f}")

    # Check Proposition 1: is effective coherence bounded despite increasing gamma?
    ecoh_vals = [r['eff_coherentie'] for r in results_opposed]
    ceiling = max(ecoh_vals)
    print(f"\nProposition 1 test: Eff. coherence ceiling = {ceiling:.3f}")
    print(f"  gamma at ceiling: {results_opposed[ecoh_vals.index(ceiling)]['gamma']:.2f}")
    print(f"  Even at gamma=0.80, eff. coherence = {ecoh_vals[-1]:.3f}")
    print(f"  Bounded? {'YES -- ceiling exists despite increasing gamma' if ceiling < 0.8 else 'PARTIAL -- ceiling is high'}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    gammas = [r['gamma'] for r in results_opposed]
    ax1.plot(gammas, ecoh_vals, 'o-', color='#e74c3c', label='phi opposes d (actual PO)')
    ax1.axhline(y=ceiling, color='red', linestyle='--', alpha=0.3, label=f'Ceiling = {ceiling:.3f}')
    ax1.set_xlabel('gamma (teacher goal weight)')
    ax1.set_ylabel('Effective Coherence')
    ax1.set_title('Proposition 1: gamma-Bounded Effective Coherence')
    ax1.legend()

    strains = [r['leraar_strain'] for r in results_opposed]
    ax2.plot(gammas, strains, 's-', color='#8e44ad')
    ax2.set_xlabel('gamma (teacher goal weight)')
    ax2.set_ylabel('Teacher Strain (distance from ideal)')
    ax2.set_title('Teacher Strain vs Intrinsic Motivation')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'gamma_sensitivity.png'), dpi=150)
    plt.close()
    print(f"Figure saved: {OUT}/gamma_sensitivity.png")
    return results_opposed


# ═══════════════════════════════════════════════════
# EXPERIMENT 1.5: Dimension Sensitivity
# ═══════════════════════════════════════════════════

def experiment_dimension_sensitivity():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1.5: Dimension Sensitivity (2D vs 3D)")
    print("=" * 60)

    # 3D baseline (standard)
    sys3d = make_passend_onderwijs()
    n_att_3d, _ = run_attractor_analysis(sys3d, n_starts=12)
    diag3d = get_diagnostics(sys3d)

    # 2D version: drop Resources dimension (use only Workload, Autonomy)
    actoren_2d = [
        Actor("Leraar", np.array([4.0, 6.5]), '#27ae60'),
        Actor("School", np.array([4.5, 5.5]), '#f39c12'),
        Actor("SWV", np.array([4.0, 5.0]), '#9b59b6'),
        Actor("Ministerie", np.array([3.5, 6.0]), '#e74c3c'),
    ]
    U_2d = {
        "Leraar": {
            'doel': np.array([3.0, 7.5]),
            'gewicht': 0.4,
            'alpha': {'School': -0.02, 'SWV': -0.03, 'Ministerie': -0.05}
        },
        "School": {
            'doel': np.array([5.0, 5.0]),
            'gewicht': 0.4,
            'alpha': {'Leraar': 0.05, 'SWV': 0.02, 'Ministerie': 0.05}
        },
        "SWV": {
            'doel': np.array([5.0, 6.0]),
            'gewicht': 0.3,
            'alpha': {'Leraar': 0.0, 'School': 0.05, 'Ministerie': 0.08}
        },
        "Ministerie": {
            'doel': np.array([6.0, 4.0]),
            'gewicht': 0.5,
            'alpha': {'Leraar': 0.0, 'School': 0.02, 'SWV': 0.05}
        },
    }
    W = np.array([
        [0.0, 0.5, 0.2, 0.3],
        [0.1, 0.0, 0.3, 0.4],
        [0.0, 0.2, 0.0, 0.4],
        [0.0, 0.1, 0.1, 0.0],
    ])
    C_2d = {
        "Leraar": {0: (0, 10), 1: (0, 10)},
        "School": {0: (0, 10), 1: (0, 10)},
        "SWV": {0: (0, 10), 1: (0, 10)},
        "Ministerie": {0: (0, 10), 1: (0, 10)},
    }

    sys2d = Krachtenveld(actoren_2d, U_2d, W, C=C_2d, eta=0.01)
    n_att_2d, _ = run_attractor_analysis(sys2d, n_starts=12)
    d_2d = np.array([-1, 1]) / np.sqrt(2)
    diag2d = diagnose(sys2d, d_gewenst=d_2d, test_attractoren=True)

    print(f"\n{'Dimension':>10} | {'Attractors':>10} | {'Asymmetry':>10} | {'Eff.Coh':>8}")
    print("-" * 45)
    print(f"{'2D':>10} | {n_att_2d:>10} | {diag2d.asymmetrie_W:>10.3f} | {diag2d.effectieve_coherentie:>8.3f}")
    print(f"{'3D':>10} | {n_att_3d:>10} | {diag3d['asymmetrie_W']:>10.3f} | {diag3d['eff_coherentie']:>8.3f}")

    agree = abs(n_att_2d - n_att_3d) <= 2
    print(f"\nCore finding robust across dimensions? {'YES' if agree else 'PARTIAL'}")
    print(f"  2D attractors: {n_att_2d}, 3D attractors: {n_att_3d}")
    return {'2d': n_att_2d, '3d': n_att_3d}


# ═══════════════════════════════════════════════════
# EXPERIMENT 1.7: Multi-Expert Configuration Test
# ═══════════════════════════════════════════════════

def experiment_multi_expert():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1.7: Multi-Expert Configuration Test")
    print("=" * 60)

    # Expert A: Slightly different actor positions (random +/-0.5)
    def expert_A():
        s = make_passend_onderwijs()
        for a in s.actoren:
            a.positie = a.positie + np.random.uniform(-0.5, 0.5, 3)
            a.positie = np.clip(a.positie, 0.0, 10.0)
        return s

    # Expert B: Different W structure (inspectorate=SWV more powerful)
    def expert_B():
        s = make_passend_onderwijs()
        s.W[2, :] *= 1.5  # SWV exerts more influence
        s.W[:, 2] *= 0.7  # Others influence SWV less
        return s

    # Expert C: Different ideals (teacher ideal less extreme)
    def expert_C():
        s = make_passend_onderwijs()
        s.U_config["Leraar"]["doel"] = np.array([3.5, 7.0, 6.5])  # closer to center
        return s

    # Expert D: More conflict (deeper alpha)
    def expert_D():
        return make_passend_onderwijs(alpha_leraar_min=-0.08,
                                       alpha_leraar_swv=-0.05,
                                       alpha_leraar_school=-0.04)

    # Expert E: More bottom-up influence in W
    def expert_E():
        s = make_passend_onderwijs()
        s.W[3, 0] = 0.15  # Ministry listens more to teachers
        s.W[3, 1] = 0.15  # Ministry listens more to schools
        s.W[0, 3] = 0.15  # Teachers less pushed by ministry
        return s

    experts = {
        'Baseline': lambda: make_passend_onderwijs(),
        'Expert A (positions +/-0.5)': expert_A,
        'Expert B (strong SWV)': expert_B,
        'Expert C (moderate teacher ideal)': expert_C,
        'Expert D (deeper conflict)': expert_D,
        'Expert E (bottom-up W)': expert_E,
    }

    results = {}
    print(f"\n{'Expert':>35} | {'Attractors':>10} | {'Eff.Coh':>8} | {'Asymmetry':>10} | {'Coop':>8}")
    print("-" * 85)

    for name, factory in experts.items():
        atts = []
        ecoh = []
        asym = []
        coop = []
        for _ in range(3):
            s = factory()
            n_att, _ = run_attractor_analysis(s, n_starts=12)
            d = get_diagnostics(s)
            atts.append(n_att)
            ecoh.append(d['eff_coherentie'])
            asym.append(d['asymmetrie_W'])
            coop.append(d['samenwerking'])

        results[name] = {
            'attractors': np.mean(atts),
            'eff_coherentie': np.mean(ecoh),
            'asymmetrie': np.mean(asym),
            'samenwerking': np.mean(coop),
        }
        print(f"{name:>35} | {np.mean(atts):>10.1f} | {np.mean(ecoh):>8.3f} | {np.mean(asym):>10.3f} | {np.mean(coop):>8.3f}")

    # Agreement analysis
    all_atts = [r['attractors'] for r in results.values()]
    all_multi = [a > 2 for a in all_atts]
    agreement = sum(all_multi) / len(all_multi)

    print(f"\nAgreement on multi-stability: {agreement*100:.0f}% of experts produce >2 attractors")
    print(f"Attractor range: {min(all_atts):.0f} to {max(all_atts):.0f}")
    print(f"Core finding robust? {'YES' if agreement >= 0.8 else 'PARTIAL' if agreement >= 0.5 else 'NO'}")
    return results


# ═══════════════════════════════════════════════════
# RUN ALL EXPERIMENTS
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print("SSRN PAPER — SENSITIVITY ANALYSIS")
    print("=" * 60)

    r1 = experiment_W_perturbation()
    r2 = experiment_alpha_sweep()
    r3 = experiment_gamma_sensitivity()
    r4 = experiment_dimension_sensitivity()
    r5 = experiment_multi_expert()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved in: {OUT}/")
    print("  - W_perturbation.png")
    print("  - alpha_sweep.png")
    print("  - gamma_sensitivity.png")
