import jax
import jax.numpy as jnp
import time

# 1. Pure scalar step function for a single oscillator
def oscillator_step(x, v, w, dt=0.01, c=0.1):
    """
    Pure mathematical function updating state variables for a single oscillator.
    """
    # Damped harmonic oscillator acceleration: a = -w^2 * x - c * v
    a = - (w ** 2) * x - c * v
    # Explicit Euler integration
    x_new = x + v * dt
    v_new = v + a * dt
    return x_new, v_new

# 2. Parallelize the step function across the batch using vmap
# We batch along the first dimension (axis 0) for x, v, and w
oscillator_step_batched = jax.vmap(oscillator_step, in_axes=(0, 0, 0))

# 3. Wrap the outer simulation loop in JIT compilation
@jax.jit
def run_jax_simulation(x_init, v_init, w, steps=1000):
    """
    Runs the simulation over a given number of steps using JIT compilation.
    The loop is traced and compiled into fused XLA operations on the accelerator.
    """
    x = x_init
    v = v_init
    
    # Static Python loop. JAX's JIT compiler traces and unrolls this static loop
    for _ in range(steps):
        x, v = oscillator_step_batched(x, v, w)
    return x, v

if __name__ == "__main__":
    # Initialize JAX PRNG key
    key = jax.random.PRNGKey(42)
    N = 100000
    steps = 1000
    
    # Initialize states as JAX device arrays
    x_init = jnp.ones(N, dtype=jnp.float32)
    v_init = jnp.zeros(N, dtype=jnp.float32)
    
    # Draw w from uniform distribution U(0.5, 2.0) using JAX PRNG
    key, subkey = jax.random.split(key)
    w = jax.random.uniform(subkey, shape=(N,), minval=0.5, maxval=2.0, dtype=jnp.float32)
    
    print(f"Starting JAX swarm simulation of {N} pendulums...")
    
    # 4. Tracing phase (First run includes compilation + execution)
    start_compile = time.time()
    x_compiled, v_compiled = run_jax_simulation(x_init, v_init, w, steps=steps)
    # JAX uses asynchronous dispatch, so we block until the computation completes
    x_compiled.block_until_ready()
    v_compiled.block_until_ready()
    compile_and_run_time = time.time() - start_compile
    print(f"First run (Tracing + JIT Compilation): {compile_and_run_time:.6f} seconds")
    
    # 5. Compiled Execution Phase (Second run uses the precompiled machine binary)
    start_compiled = time.time()
    x_fast, v_fast = run_jax_simulation(x_init, v_init, w, steps=steps)
    x_fast.block_until_ready()
    v_fast.block_until_ready()
    compiled_run_time = time.time() - start_compiled
    print(f"Second run (Pure Compiled JIT): {compiled_run_time:.6f} seconds")
