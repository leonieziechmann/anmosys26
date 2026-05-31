import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Total points to generate
    N = 5_000_000
    
    # 1. Measure the exact wall-clock execution time of the generation and check process
    start_time = time.perf_counter()
    
    # Generate points uniformly distributed between 0 and 1
    x = np.random.uniform(0.0, 1.0, N)
    y = np.random.uniform(0.0, 1.0, N)
    
    # Compute squared Euclidean distance and check inside unit circle boundary
    squared_distance = x**2 + y**2
    inside_mask = squared_distance <= 1.0
    
    execution_time = time.perf_counter() - start_time
    
    # Count points inside and calculate empirical Pi
    N_inside = np.sum(inside_mask)
    pi_estimate = 4.0 * N_inside / N
    
    print(f"Classical NumPy Pi Estimation:")
    print(f"  Total samples (N): {N:,}")
    print(f"  Samples inside:    {N_inside:,}")
    print(f"  Estimated Pi:      {pi_estimate:.6f}")
    print(f"  Execution Time:    {execution_time:.6f} seconds")
    
    # 2. Extract a random subset of 10,000 points for visualization
    subset_size = 10_000
    sub_x = x[:subset_size]
    sub_y = y[:subset_size]
    sub_inside = inside_mask[:subset_size]
    
    # Create high-quality visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
    # Scatter points inside (blue) and outside (red)
    ax.scatter(sub_x[sub_inside], sub_y[sub_inside], color='#1e88e5', s=2, alpha=0.7, label='Innerhalb (Inside)')
    ax.scatter(sub_x[~sub_inside], sub_y[~sub_inside], color='#e53935', s=2, alpha=0.7, label='Außerhalb (Outside)')
    
    # Plot the quarter-circle boundary line
    theta = np.linspace(0, np.pi/2, 200)
    cx = np.cos(theta)
    cy = np.sin(theta)
    ax.plot(cx, cy, color='#263238', linewidth=2.5, linestyle='-', label='Einheitskreisrand ($x^2+y^2=1$)')
    
    # Label and title axes
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('x-Koordinate', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('y-Koordinate', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'Klassische Monte-Carlo-Simulation zur $\\pi$-Bestimmung\nSchätzung: $\\pi \\approx$ {pi_estimate:.6f} | Laufzeit: {execution_time:.4f}s', 
                 fontsize=13, fontweight='bold', pad=15)
    
    # Aspect ratio and grids
    ax.set_aspect('equal')
    ax.legend(loc='upper right', frameon=True, shadow=False, facecolor='white', edgecolor='#cfd8dc', framealpha=0.9, fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#cfd8dc')
    
    # Save the output figure to the data directory
    plt.tight_layout()
    output_path = '/home/xayah/Documents/anmosys26/data/classical_pi_disp.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to: {output_path}")

if __name__ == '__main__':
    main()
