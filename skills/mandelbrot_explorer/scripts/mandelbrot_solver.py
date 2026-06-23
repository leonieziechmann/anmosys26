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

def simulate_mandelbrot(center_real: float, center_imag: float, zoom: float, max_iterations: int = 500) -> dict:
    """
    Runs a JAX-accelerated Mandelbrot simulation on the specified center coordinates and zoom factor.
    Returns visual complexity and Shannon entropy metrics.
    """
    _, metrics = run_simulation(center_real, center_imag, zoom, max_iterations=max_iterations)
    return metrics
