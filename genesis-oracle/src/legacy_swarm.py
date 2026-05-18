import numpy as np
import time

def run_legacy_simulation(x_init, v_init, w, dt=0.01, c=0.1, steps=1000):
    """
    Simulates a large kinetic energy harvester array (damped harmonic oscillators)
    using sequential Python loops and vectorized numpy operations.
    """
    x = x_init.copy()
    v = v_init.copy()
    
    # Start the stopwatch
    start_time = time.time()
    
    for _ in range(steps):
        # Damped harmonic oscillator acceleration: a = -w^2 * x - c * v
        a = - (w ** 2) * x - c * v
        # Explicit Euler integration
        x = x + v * dt
        v = v + a * dt
        
    # Stop the stopwatch
    end_time = time.time()
    elapsed = end_time - start_time
    return x, v, elapsed

if __name__ == "__main__":
    np.random.seed(42)
    N = 100000
    steps = 1000
    
    # Initialize the states of the 100,000 pendulums
    x_init = np.ones(N, dtype=np.float32)
    v_init = np.zeros(N, dtype=np.float32)
    
    # Initialize random natural frequencies w
    w = np.random.uniform(0.5, 2.0, N).astype(np.float32)
    
    print(f"Starting legacy swarm simulation of {N} pendulums...")
    x, v, elapsed = run_legacy_simulation(x_init, v_init, w, steps=steps)
    print("Simulation completed successfully.")
    print(f"Total Execution Time (Legacy Numpy): {elapsed:.6f} seconds")
