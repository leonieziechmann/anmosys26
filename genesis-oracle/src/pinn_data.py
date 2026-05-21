"""
pinn_data.py - Mesh-free sampling for PINN 1D Heat Equation.

Part of Angewandte Modellierung und Systemsimulation SoSe2026.
Problem Set 5: Project Genesis - The Fabric of Reality.
"""

import jax
import jax.numpy as jnp

def generate_pinn_data(key, num_col=5000, num_ic=500, num_bc=500):
    """
    Generates mesh-free sampling points for the 1D Heat Equation.
    
    Spatial domain: x in [-1, 1]
    Temporal domain: t in [0, 1]
    
    Parameters:
        key: jax.random.PRNGKey for deterministic generation
        num_col: number of collocation points (PDE residual evaluation)
        num_ic: number of initial condition points
        num_bc: number of boundary condition points
        
    Returns:
        dict: containing collocation, IC, and BC tensors
    """
    # Split random keys to maintain deterministic chaos
    key_col_x, key_col_t, key_ic, key_bc = jax.random.split(key, 4)
    
    # 1. Collocation Points (PDE evaluation points in domain interior)
    x_col = jax.random.uniform(key_col_x, (num_col,), minval=-1.0, maxval=1.0)
    t_col = jax.random.uniform(key_col_t, (num_col,), minval=0.0, maxval=1.0)
    
    # 2. Initial Condition Points (t = 0, u(x, 0) = -sin(pi * x))
    x_ic = jax.random.uniform(key_ic, (num_ic,), minval=-1.0, maxval=1.0)
    t_ic = jnp.zeros((num_ic,))
    u_ic = -jnp.sin(jnp.pi * x_ic)
    
    # 3. Boundary Condition Points (x = -1 or x = 1, t in [0, 1], u(+-1, t) = 0)
    # Generate 50% left boundary (x = -1) and 50% right boundary (x = 1)
    num_left = num_bc // 2
    num_right = num_bc - num_left
    
    x_bc = jnp.concatenate([jnp.ones(num_left) * -1.0, jnp.ones(num_right) * 1.0])
    t_bc = jax.random.uniform(key_bc, (num_bc,), minval=0.0, maxval=1.0)
    u_bc = jnp.zeros((num_bc,))
    
    return {
        "col": {"x": x_col, "t": t_col},
        "ic": {"x": x_ic, "t": t_ic, "u": u_ic},
        "bc": {"x": x_bc, "t": t_bc, "u": u_bc}
    }

if __name__ == "__main__":
    # Deterministic generation check
    master_key = jax.random.PRNGKey(42)
    data = generate_pinn_data(master_key)
    
    print("--- Mesh-Free Dataset Initialization ---")
    print(f"Collocation points (PDE): x shape = {data['col']['x'].shape}, t shape = {data['col']['t'].shape}")
    print(f"Initial conditions (IC):  x shape = {data['ic']['x'].shape}, t shape = {data['ic']['t'].shape}, u shape = {data['ic']['u'].shape}")
    print(f"Boundary conditions (BC): x shape = {data['bc']['x'].shape}, t shape = {data['bc']['t'].shape}, u shape = {data['bc']['u'].shape}")
    
    # Sanity checks
    assert jnp.all(data['ic']['t'] == 0.0), "IC time coordinates must be exactly 0!"
    assert jnp.all((data['bc']['x'] == -1.0) | (data['bc']['x'] == 1.0)), "BC spatial coordinates must be at boundary edges!"
    assert jnp.all(data['bc']['u'] == 0.0), "Dirichlet BCs should have temperature u = 0!"
    print("\nDeterministic verification checks: PASSED.")
