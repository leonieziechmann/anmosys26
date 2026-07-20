import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_signals import inject_telemetry_disturbance

def simulate_damped_oscillator(t, beta, omega_0, x0=1.0, v0=0.0):
    """
    Simulates a damped harmonic oscillator using explicit Euler integration.
    Equation: d2x/dt2 + 2*beta*dx/dt + w0^2*x = 0
    """
    dt = t[1] - t[0]
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    x[0] = x0
    v[0] = v0
    
    for i in range(1, len(t)):
        a = -2 * beta * v[i-1] - (omega_0 ** 2) * x[i-1]
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i-1] * dt
    return x

def main():
    print("Initializing Physics Data-Feed & Plotting (Damped Harmonic Oscillator)...")
    
    # Time vector (10 seconds, 1000 points)
    t = np.linspace(0.0, 10.0, 1000)
    
    # Parameters
    beta = 0.15        # Initial weak damping factor (Agent A's first guess)
    omega_0 = np.pi    # Fundamental frequency (w0 = pi)
    
    # 1. Run simulation
    signal = simulate_damped_oscillator(t, beta, omega_0)
    
    # 2. Inject disturbance/stochastic noise around t = 4.25s
    perturbed_signal = inject_telemetry_disturbance(t, signal.copy())
    
    # 3. Create a high-quality, dark-themed visualization (Premium Aesthetics)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Plot normal vs perturbed
    ax.plot(t, perturbed_signal, color='#00e5ff', linewidth=2, label='Telemetry Signal (Perturbed)')
    ax.plot(t, signal, color='#ff007f', linewidth=1.5, linestyle='--', alpha=0.6, label='Ideal Damped Oscillator')
    
    # Highlight anomaly region
    ax.axvspan(4.0, 4.5, color='#ff1744', alpha=0.15, label='Anomaly Event Horizon')
    
    # Premium labels & styling
    ax.set_title('Telemetry Real-Time Analysis: Damped Harmonic Oscillator', fontsize=14, fontweight='bold', pad=15, color='#ffffff')
    ax.set_xlabel('Time (seconds)', fontsize=11, labelpad=10, color='#e0e0e0')
    ax.set_ylabel('Amplitude (x)', fontsize=11, labelpad=10, color='#e0e0e0')
    
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='#424242')
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 1.5)
    
    # Custom legend
    legend = ax.legend(loc='upper right', frameon=True, facecolor='#121212', edgecolor='#333333')
    plt.setp(legend.get_texts(), color='#e0e0e0')
    
    # Save the output figure
    os.makedirs('data', exist_ok=True)
    output_path = 'data/anomaly_detection_plot.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='#121212')
    plt.close()
    
    print(f"Telemetry plot exported successfully to: {output_path}")

if __name__ == '__main__':
    main()
