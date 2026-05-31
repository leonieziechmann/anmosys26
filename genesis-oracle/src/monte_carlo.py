import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Force JAX to use CPU for simplicity and environment consistency (avoiding GPU driver mismatches)
jax.config.update('jax_platform_name', 'cpu')

def simulate_path(key):
    """
    Mathematically pure JAX function to simulate a single market path.
    Consumes a single PRNGKey, splits it into three independent subkeys,
    samples D, C, and R, and returns the scalar net revenue.
    """
    # Split the key into three independent subkeys for our three stochastic variables
    key_D, key_C, key_R = jax.random.split(key, 3)
    
    # 1. Market Demand (D): D ~ N(1000, 150^2)
    D = 1000.0 + 150.0 * jax.random.normal(key_D)
    
    # 2. Production Asset Cost (C): ln(C) ~ N(5.5, 0.3^2)
    # C = exp(5.5 + 0.3 * normal)
    C = jnp.exp(5.5 + 0.3 * jax.random.normal(key_C))
    
    # 3. Regulatory Penalty Rate (R): R ~ U(0.05, 0.25)
    R = jax.random.uniform(key_R, minval=0.05, maxval=0.25)
    
    # 4. Net Revenue Equation
    revenue = (D * 150.0) - C * (1.0 - R)
    return revenue

def main():
    # 1. Initialize master key and split into 1,000,000 unique subkeys
    master_key = jax.random.PRNGKey(42)
    N = 1_000_000
    
    print(f"JAX Monte Carlo Business Simulation (N = {N:,} paths)")
    print("Generating 1,000,000 unique subkeys...")
    subkeys = jax.random.split(master_key, N)
    
    # 2. Vectorize the simulate_path function using jax.vmap
    simulate_paths_vmap = jax.vmap(simulate_path)
    
    # 3. First execution (Cold run: JIT-compiling & Tracing + Execution)
    print("Executing Cold Run (includes tracing and JIT compilation)...")
    start_time = time.perf_counter()
    revenues_cold = simulate_paths_vmap(subkeys)
    # Force evaluation since JAX is asynchronous
    revenues_cold.block_until_ready()
    cold_time = time.perf_counter() - start_time
    
    # 4. Second execution (Warm run: pure XLA execution from cache)
    print("Executing Warm Run (pure compiled XLA execution)...")
    start_time = time.perf_counter()
    revenues_warm = simulate_paths_vmap(subkeys)
    revenues_warm.block_until_ready()
    warm_time = time.perf_counter() - start_time
    
    # 5. Extract results (use warm run revenues, which are identical due to determinism)
    revenues = np.array(revenues_warm)
    expected_revenue = np.mean(revenues)
    
    # Value-at-Risk (VaR 95%): the 5th percentile of the sorted revenue array
    var_95 = np.percentile(revenues, 5.0)
    
    speedup = cold_time / warm_time if warm_time > 0 else 0
    tps = N / warm_time if warm_time > 0 else 0
    
    print("\nSimulation Results:")
    print(f"  Expected Revenue:  {expected_revenue:.2f} €")
    print(f"  Value-at-Risk 95%: {var_95:.2f} €")
    print(f"  Cold Run Time:     {cold_time:.6f} seconds")
    print(f"  Warm Run Time:     {warm_time:.6f} seconds")
    print(f"  Speedup Factor:    {speedup:.2f}x")
    print(f"  XLA Throughput:    {tps:,.2f} paths/second")
    
    # 6. Generate a clean distribution histogram
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Plot histogram of revenues
    counts, bins, patches = ax.hist(revenues, bins=100, color='#009688', alpha=0.8, 
                                    edgecolor='#00796b', linewidth=0.5, label='Umsatz-Verteilung (Revenue Dist)')
    
    # Draw vertical line for expected revenue
    ax.axvline(expected_revenue, color='#212121', linestyle='-', linewidth=2, 
               label=f'Erwarteter Umsatz (Expected): {expected_revenue:,.2f} €')
    
    # Draw vertical line for Value-at-Risk 95%
    ax.axvline(var_95, color='#d32f2f', linestyle='--', linewidth=2, 
               label=f'Value-at-Risk (VaR 95%): {var_95:,.2f} €')
    
    # Shade the 5% risk region (below VaR 95%) in light red
    for patch, left, right in zip(patches, bins[:-1], bins[1:]):
        if right <= var_95:
            patch.set_facecolor('#ef5350')
            patch.set_edgecolor('#d32f2f')
            
    ax.set_xlabel('Netto-Jahresumsatz (Net Revenue) [€]', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Häufigkeit (Frequency)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(f'JAX-beschleunigte Monte-Carlo-Umsatzsimulation (N = 1.000.000)\n'
                 f'Erwarteter Umsatz: {expected_revenue:,.2f} € | VaR 95%: {var_95:,.2f} €', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Format tick labels with thousand separator
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, loc: "{:,}".format(int(y))))
    
    ax.legend(loc='upper right', frameon=True, shadow=False, facecolor='white', edgecolor='#cfd8dc', framealpha=0.9, fontsize=9.5)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#cfd8dc')
    
    # Save the resulting graphic as data/revenue_dist.png
    plt.tight_layout()
    output_path = '/home/xayah/Documents/anmosys26/data/revenue_dist.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nDistribution histogram saved to: {output_path}")

if __name__ == '__main__':
    main()
