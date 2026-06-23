import jax
import jax.numpy as jnp

# JAX Mandelbrot Core Kernel - Fixed Loop Condition using loop counter i
@jax.jit
def mandelbrot_kernel(c, max_iters):
    def body_fn(val):
        z, count, active, i = val
        next_z = z**2 + c
        next_active = active & (jnp.abs(next_z) <= 2.0)
        next_count = jnp.where(next_active, count + 1, count)
        return next_z, next_count, next_active, i + 1

    def cond_fn(val):
        _, _, active, i = val
        return jnp.any(active) & (i < max_iters)

    z = jnp.zeros_like(c)
    count = jnp.zeros_like(c, dtype=jnp.int32)
    active = jnp.ones_like(c, dtype=jnp.bool_)
    
    _, final_counts, _, _ = jax.lax.while_loop(cond_fn, body_fn, (z, count, active, 0))
    return final_counts

def run_simulation(center_real, center_imag, zoom, resolution=400, max_iterations=500):
    # Map resolution coordinates to complex plane
    width, height = resolution, resolution
    r = jnp.linspace(center_real - 1.5 / zoom, center_real + 1.5 / zoom, width)
    i = jnp.linspace(center_imag - 1.5 / zoom, center_imag + 1.5 / zoom, height)
    R, I = jnp.meshgrid(r, i)
    C = R + 1j * I

    # Execute high-performance calculation on accelerator
    counts = mandelbrot_kernel(C.flatten(), max_iterations)
    counts = counts.reshape((height, width))

    # Calculate Shannon Entropy of escape times
    hist, _ = jnp.histogram(counts, bins=20)
    hist_prob = hist / jnp.sum(hist)
    hist_prob = jnp.where(hist_prob > 0, hist_prob, 1.0) # Avoid log(0)
    entropy = -jnp.sum(hist_prob * jnp.log(hist_prob))

    # Calculate Boundary Complexity (ratio of boundary pixels)
    boundary_pixels = jnp.sum((counts > 0) & (counts < max_iterations))
    boundary_ratio = boundary_pixels / (width * height)

    return counts, {
        "entropy": float(entropy),
        "boundary_complexity": float(boundary_ratio),
        "center_real": float(center_real),
        "center_imag": float(center_imag),
        "zoom": float(zoom),
        "max_iterations": int(max_iterations)
    }

def main():
    print("--- STEP 0: Global View ---")
    _, metrics0 = run_simulation(center_real=-0.5, center_imag=0.0, zoom=1.5)
    print("Metrics Step 0:", metrics0)

    # Manual refinement steps towards Seahorse Valley c = -0.7436 + 0.1318i
    print("\n--- STEP 1: Zooming in towards c = -0.7436 + 0.1318i (Zoom 10x) ---")
    _, metrics1 = run_simulation(center_real=-0.74, center_imag=0.13, zoom=10.0)
    print("Metrics Step 1:", metrics1)

    print("\n--- STEP 2: Zooming in closer (Zoom 100x) ---")
    _, metrics2 = run_simulation(center_real=-0.743, center_imag=0.131, zoom=100.0)
    print("Metrics Step 2:", metrics2)

    print("\n--- STEP 3: Zooming in closer (Zoom 1000x) ---")
    _, metrics3 = run_simulation(center_real=-0.7436, center_imag=0.1318, zoom=1000.0)
    print("Metrics Step 3:", metrics3)

if __name__ == "__main__":
    main()
