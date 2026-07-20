import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

def inject_telemetry_disturbance(t, signal):
    """
    Injects a high-frequency stochastic noise/disturbance in the datastream around t = 4.25s.
    """
    disturbance_mask = (t >= 4.0) & (t <= 4.5)
    noise = 1.5 * np.sin(2 * np.pi * 50 * t[disturbance_mask]) * np.random.normal(1.0, 0.2, np.sum(disturbance_mask))
    signal[disturbance_mask] += noise
    return signal

def main():
    # Setup directories
    os.makedirs("data", exist_ok=True)
    
    # 1. Generate normal wave signal
    t = np.linspace(0, 10, 1000)
    # Base signal: combination of sine waves
    signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.cos(2 * np.pi * 1.5 * t)
    
    # 2. Inject high-frequency clipping artifact (amplitude saturation)
    # Choose a random start index for the anomaly
    anomaly_start = random.randint(600, 800)
    anomaly_duration = 40
    anomaly_end = anomaly_start + anomaly_duration
    
    # Inject high-frequency noise/wave
    signal[anomaly_start:anomaly_end] += 3.0 * np.sin(2 * np.pi * 50 * t[anomaly_start:anomaly_end])
    
    # Apply amplitude saturation (clipping) to the entire signal
    clipped_signal = np.clip(signal, -1.2, 1.2)
    
    # Save the anomaly index secretly to a JSON file so the mock client can detect it
    # without printing to the terminal
    info_path = os.path.join("data", "anomaly_info.json")
    with open(info_path, "w") as f:
        json.dump({"clipping_index": anomaly_start}, f)
        
    # 3. Plot and save
    plt.figure(figsize=(10, 5))
    plt.plot(t, clipped_signal, label="Telemetry Signal", color="#00d2ff")
    plt.title("System Telemetry Signal", fontsize=14, color="white")
    plt.xlabel("Time (s)", fontsize=12, color="white")
    plt.ylabel("Amplitude", fontsize=12, color="white")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(facecolor="#1e1e1e", edgecolor="#00d2ff", labelcolor="white")
    
    # Dark themed plot
    fig = plt.gcf()
    ax = plt.gca()
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    plot_path = os.path.join("data", "audit_target.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print(f"Wave signal generated. Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
