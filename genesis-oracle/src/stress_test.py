import jax
import jax.numpy as jnp
import numpy as np

# Force JAX to use CPU
jax.config.update('jax_platform_name', 'cpu')

def simulate_path_with_sigma(key, sigma_C):
    """
    Stochastic model path with adjustable log-normal standard deviation parameter (sigma_C).
    """
    key_D, key_C, key_R = jax.random.split(key, 3)
    
    # 1. Market Demand (D): D ~ N(1000, 150^2)
    D = 1000.0 + 150.0 * jax.random.normal(key_D)
    
    # 2. Production Asset Cost (C): ln(C) ~ N(5.5, sigma_C^2)
    C = jnp.exp(5.5 + sigma_C * jax.random.normal(key_C))
    
    # 3. Regulatory Penalty Rate (R): R ~ U(0.05, 0.25)
    R = jax.random.uniform(key_R, minval=0.05, maxval=0.25)
    
    # 4. Net Revenue Equation
    revenue = (D * 150.0) - C * (1.0 - R)
    return revenue

def run_simulation(master_key, sigma_C, N=1_000_000):
    """
    Generates N subkeys, vectorizes path simulation with sigma_C,
    and returns expected revenue and VaR 95%.
    """
    subkeys = jax.random.split(master_key, N)
    simulate_paths_vmap = jax.vmap(lambda k: simulate_path_with_sigma(k, sigma_C))
    revenues = simulate_paths_vmap(subkeys)
    # block until ready to ensure accurate timing/execution
    revenues.block_until_ready()
    
    revenues_np = np.array(revenues)
    expected_revenue = np.mean(revenues_np)
    var_95 = np.percentile(revenues_np, 5.0)
    return expected_revenue, var_95

def main():
    master_key = jax.random.PRNGKey(42)
    
    print("Subagent-Alpha ('The Stress-Tester') active.")
    print("Initiating automated parameter sweep on Log-Normal asset cost volatility...")
    print(f"{'Cost Sigma (σ_C)':<18} | {'Variance (σ_C^2)':<18} | {'Expected Revenue (€)':<22} | {'VaR 95% (€)':<18}")
    print("-" * 85)
    
    # Linear sweep from 0.3 to 6.0 with step 0.3
    sigmas = np.arange(0.3, 6.01, 0.3)
    
    breaking_point = None
    results = []
    
    for sigma in sigmas:
        variance = sigma ** 2
        expected_revenue, var_95 = run_simulation(master_key, sigma)
        results.append((sigma, variance, expected_revenue, var_95))
        print(f"{sigma:<18.2f} | {variance:<18.4f} | {expected_revenue:<22.2f} | {var_95:<18.2f}")
        
        # Identify first crossing point
        if var_95 < 0 and breaking_point is None:
            breaking_point = sigma
            
    print("-" * 85)
    
    # Fine-grain binary search or linear sweep to find the exact breaking point to 2 decimal places
    if breaking_point is not None:
        print(f"Detected risk transition window. Running fine-grid sweep around σ_C = {breaking_point:.2f}...")
        fine_sigmas = np.arange(breaking_point - 0.30, breaking_point + 0.01, 0.01)
        
        exact_sigma = None
        for fs in fine_sigmas:
            exp_rev, v95 = run_simulation(master_key, fs)
            if v95 < 0:
                exact_sigma = fs
                break
        
        if exact_sigma is None:
            exact_sigma = breaking_point
            
        print(f"\n[STRESS TEST RESULT] Breaking Point Located:")
        print(f"  Critical σ_C: {exact_sigma:.2f}")
        print(f"  Critical Variance (σ_C^2): {exact_sigma**2:.4f}")
        print(f"  At this threshold, VaR 95% turns negative, indicating that tail risk exceeds enterprise revenue.")

if __name__ == '__main__':
    main()
