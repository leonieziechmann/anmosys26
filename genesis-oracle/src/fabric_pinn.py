"""
fabric_pinn.py - Physics-Informed Neural Network (PINN) for the 1D Heat Equation.

Part of Angewandte Modellierung und Systemsimulation SoSe2026.
Problem Set 5: Project Genesis - The Fabric of Reality.
"""

import sys
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm

# Ensure local imports work correctly regardless of launch context
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pinn_data import generate_pinn_data

# ==============================================================================
# 1. Neural Surrogate Architecture (Flax Linen)
# ==============================================================================

class HeatSurrogate(nn.Module):
    """
    Multi-Layer Perceptron representing the continuous spacetime temperature field u(x, t).
    
    Architecture:
        - Input: 2D coordinate (x, t)
        - Hidden layers: 4 dense layers, 32 neurons each
        - Activation function: tanh (smooth, non-zero derivatives)
        - Output: 1D scalar temperature u
    """
    @nn.compact
    def __call__(self, coords):
        # coords can be of shape (2,) for single point or (N, 2) for batch
        x = nn.Dense(32)(coords)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x

# Initialize global model
model = HeatSurrogate()

# ==============================================================================
# 2. Physics Engine & Unified Gradient Setup
# ==============================================================================

# Thermal diffusivity parameter
alpha = 0.05

def predict_single_u(params, x, t):
    """
    Evaluates the continuous field u(x, t) at a single coordinate point.
    Returns a scalar JAX value, allowing nested automatic differentiation.
    """
    inp = jnp.stack([x, t]) # shape (2,)
    pred = model.apply({'params': params}, inp) # shape (1,)
    return pred[0]

# Exact analytical derivatives using automatic differentiation (Autodiff)
# u_t: first derivative with respect to time (t)
u_t_fn = jax.grad(predict_single_u, argnums=2)

# u_x: first derivative with respect to space (x)
u_x_fn = jax.grad(predict_single_u, argnums=1)

# u_xx: second derivative with respect to space (x)
u_xx_fn = jax.grad(u_x_fn, argnums=1)

def pde_residual_single(params, x, t):
    """
    Calculates the exact PDE residual: u_t - alpha * u_xx
    For a valid physical solution, this residual should be zero everywhere.
    """
    u_t = u_t_fn(params, x, t)
    u_xx = u_xx_fn(params, x, t)
    return u_t - alpha * u_xx

# Vectorized operations for parallel evaluation across the collocation batch
predict_batch_u = jax.vmap(predict_single_u, in_axes=(None, 0, 0))
pde_residual_batch = jax.vmap(pde_residual_single, in_axes=(None, 0, 0))

# ==============================================================================
# 3. Optimization & Training Pipeline (Optax & JIT Compilation)
# ==============================================================================

def train_pinn(data, epochs=10000, lr=1e-3, seed=42):
    """
    Compiles and executes the JAX-accelerated training pipeline.
    """
    # 1. Initialize Model weights explicitly
    key = jax.random.PRNGKey(seed)
    key_init, key_opt = jax.random.split(key)
    dummy_input = jnp.zeros((1, 2))
    variables = model.init(key_init, dummy_input)
    params = variables['params']
    
    # 2. Setup Optax optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    
    # 3. JIT-compiled optimization step
    @jax.jit
    def train_step(p, o_state, d):
        def loss_fn(current_params):
            # PDE Collocation Loss (Physics constraints)
            residuals = pde_residual_batch(current_params, d['col']['x'], d['col']['t'])
            physics_loss = jnp.mean(residuals**2)
            
            # Initial Condition Loss (IC genesis anchor)
            u_ic_pred = predict_batch_u(current_params, d['ic']['x'], d['ic']['t'])
            ic_loss = jnp.mean((u_ic_pred - d['ic']['u'])**2)
            
            # Boundary Condition Loss (BC limits sandbox)
            u_bc_pred = predict_batch_u(current_params, d['bc']['x'], d['bc']['t'])
            bc_loss = jnp.mean((u_bc_pred - d['bc']['u'])**2)
            
            # Unified Total Loss
            total_loss = physics_loss + ic_loss + bc_loss
            return total_loss, (physics_loss, ic_loss, bc_loss)
            
        # Compute value & gradients
        (loss, (p_loss, i_loss, b_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        
        # Apply optimizer updates
        updates, next_o_state = optimizer.update(grads, o_state, p)
        next_params = optax.apply_updates(p, updates)
        
        return next_params, next_o_state, loss, p_loss, i_loss, b_loss

    print("Transpiling mathematical derivatives into fused XLA kernels...")
    print("Initiating Silicon Ignition (GPU/CPU compilation & optimization loop)...")
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        params, opt_state, total_loss, p_loss, i_loss, b_loss = train_step(params, opt_state, data)
        
        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Total Loss: {total_loss:.6e} | PDE Loss: {p_loss:.6e} | IC Loss: {i_loss:.6e} | BC Loss: {b_loss:.6e}")
            
    print("Silicon training loop complete. The surrogate has internalized thermodynamics.")
    return params

# ==============================================================================
# 4. Meshgrid Prediction & Visualizations
# ==============================================================================

def visualize_solution(params, project_root_path):
    """
    Generates meshgrid predictions and renders beautiful 3D visualizations.
    """
    # Create continuous prediction meshgrid (100 x 100 points)
    x_grid = jnp.linspace(-1.0, 1.0, 100)
    t_grid = jnp.linspace(0.0, 1.0, 100)
    X, T = jnp.meshgrid(x_grid, t_grid)
    
    # Flat arrays for fast JAX vectorized execution
    X_flat = X.ravel()
    T_flat = T.ravel()
    U_flat = predict_batch_u(params, X_flat, T_flat)
    U = U_flat.reshape(X.shape)
    
    # Convert back to NumPy for standard plotting libraries
    X_np = np_x = np.array(X)
    T_np = np_t = np.array(T)
    U_np = np_u = np.array(U)
    
    # Create output directories if needed
    data_dir = os.path.join(project_root_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # A. Render Interactive Plotly Visualization
    print("Weaving interactive 3D Spacetime surface tensor in Plotly...")
    fig = go.Figure(data=[go.Surface(
        x=X_np, y=T_np, z=U_np, 
        colorscale='inferno',
        colorbar=dict(title="Temperature (u)", thickness=20, len=0.6)
    )])
    fig.update_layout(
        title={
            'text': "<b>The Fabric of Spacetime: PINN Heat Equation Solution</b><br><sup>Continuous coordinate representation of thermal diffusion</sup>",
            'y':0.92, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        scene=dict(
            xaxis=dict(title='Space (x)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
            yaxis=dict(title='Time (t)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
            zaxis=dict(title='Temperature (u)', backgroundcolor="rgb(20, 20, 20)", gridcolor="rgb(50, 50, 50)", showbackground=True),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
        ),
        paper_bgcolor="rgb(10, 10, 10)",
        font=dict(color="white", family="Outfit, Inter, sans-serif"),
        autosize=True,
        width=1000,
        height=800,
        margin=dict(l=50, r=50, b=50, t=100)
    )
    
    # Save standalone interactive HTML
    html_path = os.path.join(data_dir, "pinn_3d_fabric.html")
    fig.write_html(html_path)
    print(f"Successfully exported interactive HTML plot: {html_path}")
    
    # B. Render High-Resolution Static Matplotlib Plot for Markdown reports
    print("Generating striking angled static 3D screenshot...")
    fig_plt = plt.figure(figsize=(12, 9), facecolor='black', dpi=150)
    ax = fig_plt.add_subplot(111, projection='3d', facecolor='black')
    
    # Plot the surface
    surf = ax.plot_surface(
        X_np, T_np, U_np,
        cmap=cm.inferno,
        edgecolor='none',
        alpha=0.95,
        rcount=100, ccount=100
    )
    
    # Customize design system to look dark & premium
    ax.set_title('Continuous Space-Time Fabric: PINN 1D Heat Equation', color='white', fontsize=16, pad=30, fontweight='bold')
    ax.set_xlabel('Space (x)', color='silver', fontsize=11, labelpad=12)
    ax.set_ylabel('Time (t)', color='silver', fontsize=11, labelpad=12)
    ax.set_zlabel('Temperature (u)', color='silver', fontsize=11, labelpad=12)
    
    # Aesthetics for grids and panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    
    ax.grid(True, color='dimgray', linestyle='--', alpha=0.3)
    ax.tick_params(colors='silver', labelsize=9)
    
    # Colorbar
    cbar = fig_plt.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.08)
    cbar.set_label('Temperature (u)', color='silver', size=11, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color='silver', labelcolor='silver')
    
    # Striking angled perspective view
    ax.view_init(elev=25, azim=-45)
    
    png_path = os.path.join(data_dir, "pinn_3d_fabric.png")
    plt.savefig(png_path, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Successfully saved static high-res visualization: {png_path}")

# ==============================================================================
# 5. Main Execution Entry point
# ==============================================================================

if __name__ == "__main__":
    import numpy as np # Used strictly for plotly/matplotlib surface drawing conversion
    
    # Setup working folders
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir) # Documents/anmosys26/
    
    print("\n" + "="*80)
    print("                  OBSERVER-PRIME: NEURAL PHYSICS SURROGATE                   ")
    print("="*80)
    
    # Generate data using key to preserve deterministic chaos
    data_key = jax.random.PRNGKey(101)
    dataset = generate_pinn_data(data_key)
    
    # Train our PINN to solve the 1D Heat Equation
    final_params = train_pinn(dataset, epochs=10000, lr=2e-3)
    
    # Run predictions and generate visualizations
    visualize_solution(final_params, project_root)
    print("="*80 + "\n")
