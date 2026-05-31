import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Force JAX to use CPU
jax.config.update('jax_platform_name', 'cpu')

def main():
    # 1. Define transition probability matrices
    # Baseline Transition Matrix (P_baseline)
    P_baseline = jnp.array([
        [0.85, 0.12, 0.03], # State 0: Bull Market
        [0.10, 0.75, 0.15], # State 1: Stagnation
        [0.05, 0.20, 0.75]  # State 2: Catastrophic Recession
    ], dtype=jnp.float32)
    
    # Shock Transition Matrix (P_shock)
    # Both State 0 and State 1 shift 80% of their transition mass directly into State 2
    # Row 0: [0.85 * 0.20, 0.12 * 0.20, 0.03 + 0.80 * (0.85 + 0.12)] = [0.17, 0.024, 0.806]
    # Row 1: [0.10 * 0.20, 0.75 * 0.20, 0.15 + 0.80 * (0.10 + 0.75)] = [0.02, 0.15, 0.83]
    # Row 2: Unchanged [0.05, 0.20, 0.75]
    P_shock = jnp.array([
        [0.17,  0.024, 0.806],
        [0.02,  0.15,  0.83 ],
        [0.05,  0.20,  0.75 ]
    ], dtype=jnp.float32)
    
    # 2. Define the highly optimized step function for jax.lax.scan
    def step_fn(carry_v, t):
        """
        Computes state transition at step t.
        Uses jax.lax.select to conditionally load transition matrix without branching.
        """
        # Sabotage active for exactly 10 days (from day 180 to 190, inclusive/exclusive bounds)
        is_shock = (t >= 180) & (t < 190)
        
        # Select appropriate transition matrix
        P = jax.lax.select(is_shock, P_shock, P_baseline)
        
        # Update probability distribution vector: v_{t+1} = v_t * P
        next_v = jnp.dot(carry_v, P)
        
        return next_v, next_v

    # 3. Running the simulation loop using jax.lax.scan
    v_0 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32) # Starting in 100% Bull Market (State 0)
    days = jnp.arange(365, dtype=jnp.int32)
    
    print("Compiling and executing JAX Markov Chain simulation (Module Alpha - The Matrix Carrier)...")
    _, v_history = jax.lax.scan(step_fn, v_0, days)
    v_history.block_until_ready()
    
    # Prepend day 0 initial state for completeness (timeline 0 to 365)
    v_all = jnp.concatenate([v_0[None, :], v_history], axis=0)
    v_all_np = np.array(v_all) * 100.0 # Convert to percentages
    
    timeline = np.arange(366)
    
    # 4. Generate high-quality percentage distribution plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    
    # Plot state distributions over time
    ax.plot(timeline, v_all_np[:, 0], color='#2e7d32', linewidth=2.5, linestyle='-', label='Staat 0: Bullenmarkt (Bull Market)')
    ax.plot(timeline, v_all_np[:, 1], color='#f9a825', linewidth=2.5, linestyle='-', label='Staat 1: Stagnation (Stagnation)')
    ax.plot(timeline, v_all_np[:, 2], color='#c62828', linewidth=2.5, linestyle='-', label='Staat 2: Rezession (Catastrophic Recession)')
    
    # Highlight the Black Swan crisis period (Day 180 to 190)
    ax.axvspan(180, 190, color='#b0bec5', alpha=0.4, label='Schwarzer Schwan (Black Swan Shock: Tag 180-190)')
    ax.axvline(180, color='#37474f', linestyle='--', linewidth=1)
    ax.axvline(190, color='#37474f', linestyle='--', linewidth=1)
    ax.text(185, 95, 'KRISE', color='#263238', fontsize=9, fontweight='bold', ha='center', va='center', rotation=90)
    
    ax.set_xlim(-5, 370)
    ax.set_ylim(-2, 102)
    ax.set_xlabel('Simulierte Tage (Simulated Days)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Wahrscheinlichkeitsverteilung (%)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Makroökonomische Markov-Ketten-Simulation (365 Tage)\n'
                 'Verlauf der Marktstaaten unter einem temporären "Black Swan"-Rezessionsschock', 
                 fontsize=13, fontweight='bold', pad=15)
    
    ax.legend(loc='upper right', frameon=True, shadow=False, facecolor='white', edgecolor='#cfd8dc', framealpha=0.9, fontsize=9.5)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#cfd8dc')
    
    # Save the resulting graphic
    plt.tight_layout()
    output_path = '/home/xayah/Documents/anmosys26/data/markov_boss.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Markov Chain timeline plot saved to: {output_path}")
    
    # Print state snapshot after recovery (e.g., day 365)
    final_state = v_all_np[-1]
    print(f"Final State Distribution (Day 365):")
    print(f"  Bull Market:  {final_state[0]:.2f}%")
    print(f"  Stagnation:   {final_state[1]:.2f}%")
    print(f"  Recession:    {final_state[2]:.2f}%")

if __name__ == '__main__':
    main()
