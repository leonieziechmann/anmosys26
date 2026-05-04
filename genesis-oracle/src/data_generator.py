import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. KONFIGURATION (Hier deine ID eintragen!) ---
T = 02.0
C_ID = 902

# --- 2. PHYSIKALISCHE PARAMETER ---
w0 = 2 * np.pi / T # Grundfrequenz
R = 500.0          # 0.5 kOhm in Ohm
C = (1000 + C_ID) * 1e-6 # in Farad[cite: 2]

# --- 3. ZEITVEKTOR ---
# 100 Perioden[cite: 2]
periods = 100
points_per_period = 1000
t = np.linspace(0, periods * T, periods * points_per_period)

# --- 4. FOURIER-REIHE & RC-FILTER ---
y_filtered = np.zeros_like(t)

# Die ersten 9 ungeraden Harmonischen: 1, 3, 5, 7, 9, 11, 13, 15, 17[cite: 2]
harmonics = [1, 3, 5, 7, 9, 11, 13, 15, 17]

for k in harmonics:
    w_k = k * w0
    # Fourier-Koeffizient der Rechteckspannung
    amplitude_in = (4 / np.pi) * (1 / k)
    
    # Komplexe Übertragungsfunktion H(w) = 1 / (1 + j*w*R*C)[cite: 2]
    H = 1 / (1 + 1j * w_k * R * C)
    mag = np.abs(H)
    phase = np.angle(H)
    
    # Gefiltertes Signal aufsummieren
    y_filtered += amplitude_in * mag * np.sin(w_k * t + phase)

# --- 5. RAUSCHEN & SABOTAGE ---
# Gaußsches Rauschen hinzufügen[cite: 2]
noise = np.random.normal(0, 0.1, len(t))
signal_noisy = y_filtered + noise

# Sabotage: Massive hochfrequente Spannungsspitze zwischen Periode 70 und 75[cite: 2]
sabotage_mask = (t >= 70 * T) & (t <= 75 * T)
spike = 5.0 * np.sin(20 * w0 * t[sabotage_mask]) * np.random.normal(1, 0.5, np.sum(sabotage_mask))

signal_corrupted = np.copy(signal_noisy)
signal_corrupted[sabotage_mask] += spike

# --- 6. DATEN LOKAL SPEICHERN ---
# 1D kontinuierliches Array lokal im /data Ordner speichern[cite: 2]
os.makedirs('data', exist_ok=True)
np.save('data/datastream.npy', signal_corrupted)

# --- 7. PLOTTING ---
# Fenster des normalen verrauschten Signals neben der Anomalie plotten[cite: 2]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Normales Signal (z.B. Periode 10 bis 15)
normal_mask = (t >= 10 * T) & (t <= 15 * T)
ax1.plot(t[normal_mask], signal_corrupted[normal_mask], color='#1f77b4')
ax1.set_title('Normaler Datenstrom (Verrauscht)')
ax1.set_xlabel('Zeit')
ax1.set_ylabel('Amplitude')

# Plot 2: Anomalie (Periode 69 bis 76 für etwas Kontext)
anomaly_mask = (t >= 69 * T) & (t <= 76 * T) 
ax2.plot(t[anomaly_mask], signal_corrupted[anomaly_mask], color='#d62728')
ax2.set_title('Sabotage (Periode 70-75)')
ax2.set_xlabel('Zeit')
ax2.set_ylabel('Amplitude')

plt.tight_layout()

# Plot als data_feed.png speichern[cite: 2]
plt.savefig('data_feed.png')
print("Datensatz in 'data/datastream.npy' gespeichert.")
print("Plot 'data_feed.png' erfolgreich generiert.")
