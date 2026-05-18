import jax
import jax.numpy as jnp

def projectile_loss(v_initial):
    """
    Pure JAX function simulating the trajectory of a thrown object over 5 seconds.
    Uses explicit Euler integration with an air resistance drag coefficient of k = 0.5
    (chosen to ensure convergence stability under a standard 0.1 learning rate).
    
    Returns the Mean Squared Error (MSE) between the final simulated horizontal
    distance and a target distance of exactly 150.0 meters.
    """
    dt = 0.1
    t_max = 5.0
    steps = int(t_max / dt)  # 50 steps
    
    x = 0.0
    v = v_initial
    k = 0.5  # Drag coefficient
    
    # Step-by-step explicit Euler integration
    for _ in range(steps):
        x = x + v * dt
        v = v - k * v * dt
        
    target = 150.0
    loss = (x - target) ** 2
    return loss

if __name__ == "__main__":
    # Starting initial guess for the velocity
    v = 10.0
    learning_rate = 0.1
    
    # Generate the exact analytical gradient function automatically using JAX primitives
    grad_fn = jax.grad(projectile_loss)
    
    print("=== Differentiable Simulator Gradient Descent ===")
    print(f"Target Distance: 150.0 meters | Starting Guess: v = {v:.2f} m/s\n")
    
    for i in range(1, 21):
        loss_val = projectile_loss(v)
        grad_val = grad_fn(v)
        
        # Apply the gradient update: v = v - lr * grad
        v_new = v - learning_rate * grad_val
        
        print(f"Step {i:02d} | v = {v:8.4f} m/s | Loss = {loss_val:12.4f} | Gradient = {grad_val:12.4f}")
        v = v_new
        
    print(f"\nFinal Optimized Velocity: v = {v:.6f} m/s")
    
    # Verify the final physical distance reached by the optimized projectile
    x_test = 0.0
    v_test = v
    k = 0.5
    dt = 0.1
    for _ in range(50):
        x_test += v_test * dt
        v_test -= k * v_test * dt
        
    print(f"Verified Final Position: {x_test:.6f} meters (Target Error: {abs(x_test - 150.0):.6e} meters)")
