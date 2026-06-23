import os
import sys
import matplotlib.pyplot as plt
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

def run_simulation(center_real, center_imag, zoom, resolution=800, max_iterations=500):
    width, height = resolution, resolution
    r = jnp.linspace(center_real - 1.5 / zoom, center_real + 1.5 / zoom, width)
    i = jnp.linspace(center_imag - 1.5 / zoom, center_imag + 1.5 / zoom, height)
    R, I = jnp.meshgrid(r, i)
    C = R + 1j * R # Wait, this should be + 1j * I!
    # Ah! The notebook had:
    # R, I = jnp.meshgrid(r, i)
    # C = R + 1j * I
    C = R + 1j * I

    counts = mandelbrot_kernel(C.flatten(), max_iterations)
    counts = counts.reshape((height, width))
    return counts

def main():
    os.makedirs("data", exist_ok=True)
    
    # 1. Global View
    print("Generating Global View...")
    counts_global = run_simulation(center_real=-0.5, center_imag=0.0, zoom=1.5)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(counts_global, cmap='twilight_shifted', extent=[-0.5 - 1.5/1.5, -0.5 + 1.5/1.5, 0.0 - 1.5/1.5, 0.0 + 1.5/1.5])
    plt.colorbar(label='Iterations until escape')
    plt.title('Mandelbrot Global View (JAX)')
    plt.savefig('data/mandelbrot_global.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Seahorse Valley Close-Up View (15000x Zoom)
    print("Generating Seahorse Valley (15000x Zoom)...")
    center_real = -0.7436
    center_imag = 0.1318
    zoom = 15000.0
    counts_seahorse = run_simulation(center_real=center_real, center_imag=center_imag, zoom=zoom)
    
    plt.figure(figsize=(6, 6))
    extent = [center_real - 1.5/zoom, center_real + 1.5/zoom, center_imag - 1.5/zoom, center_imag + 1.5/zoom]
    plt.imshow(counts_seahorse, cmap='twilight_shifted', extent=extent)
    plt.colorbar(label='Iterations until escape')
    plt.title('Mandelbrot Seahorse Valley (15000x Zoom)')
    plt.savefig('data/mandelbrot_seahorse.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Plots generated successfully in data/")

if __name__ == "__main__":
    main()
