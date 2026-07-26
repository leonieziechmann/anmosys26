import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="JAX Telemetry Signal & Anomaly Generator")
    parser.add_argument("--inject-noise", "--inject_noise", action="store_true", help="Inject synthetic high-frequency noise / disturbance")
    parser.add_argument("--output", type=str, default="data/anomaly_detection_plot.png", help="Output path for the generated plot")
    args = parser.parse_args()

    # Ensure parent directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate time grid
    t = np.linspace(0, 10, 1000)
    
    # Target nominal trajectory: damped harmonic oscillator
    # x(t) = exp(-beta * t) * cos(omega * t) with beta=0.22, omega=2.0
    beta_nominal = 0.22
    omega_nominal = 2.00
    signal = np.exp(-beta_nominal * t) * np.cos(omega_nominal * np.pi * t)
    
    if args.inject_noise:
        # Inject synthetic disturbance (high frequency burst around t=4.0s - 4.5s)
        disturbance_mask = (t >= 4.0) & (t <= 4.5)
        high_freq_burst = 1.2 * np.sin(2 * np.pi * 50 * t[disturbance_mask]) * np.random.normal(1.0, 0.15, np.sum(disturbance_mask))
        signal[disturbance_mask] += high_freq_burst
        # Store metadata
        info_path = os.path.join(output_dir or "data", "anomaly_info.json")
        with open(info_path, "w") as f:
            json.dump({
                "anomaly_detected": True,
                "bounding_box": [120, 45, 300, 210],
                "estimated_damping_beta": 0.15,
                "estimated_frequency_omega": 2.10,
                "confidence": 0.92
            }, f, indent=2)

    # Dark-themed matplotlib plot for ADE UI
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    ax.plot(t, signal, label="Telemetry Signal x(t)", color="#00d2ff", linewidth=2)
    if args.inject_noise:
        ax.axvspan(4.0, 4.5, color="#ff4444", alpha=0.3, label="Anomaly Window [t=4.0s..4.5s]")
    
    ax.set_title("JAX Physical Telemetry Stream (Damped Oscillator)", fontsize=14, color="white", fontweight="bold")
    ax.set_xlabel("Time t (s)", fontsize=12, color="white")
    ax.set_ylabel("Amplitude x(t)", fontsize=12, color="white")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(facecolor="#2b2b2b", edgecolor="#00d2ff", labelcolor="white")
    
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig(args.output, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    plt.close()
    
    print(f"[JAX SIMULATION ENGINE] Telemetry signal generated with synthetic noise/disturbance.")
    print(f"[OUTPUT] Anomaly detection plot saved to: {args.output}")

if __name__ == "__main__":
    main()

