import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from architecture import create_windows, PhysicsAutoencoder

def main():
    # 1. Daten laden
    data_path = 'data/datastream.npy'
    if not os.path.exists(data_path):
        # Falls die Datei nicht existiert, versuchen wir sie im übergeordneten Ordner zu finden
        # oder weisen darauf hin, dass data_generator.py ausgeführt werden muss.
        print(f"Fehler: {data_path} nicht gefunden. Führe 'python src/data_generator.py' zuerst aus.")
        return

    signal = np.load(data_path)
    print(f"Signal geladen: {len(signal)} Datenpunkte")

    # 2. Windowing
    window_size = 50
    windows = create_windows(signal, window_size)
    print(f"Windows erstellt: {windows.shape}")

    # 3. Datentrennung
    # Periode 60 entspricht Index 60.000 (da 1000 Punkte pro Periode)
    # Da die Fenster überlappen, müssen wir den Splitpunkt anpassen
    split_idx = 60000 - window_size + 1
    train_data = windows[:split_idx]
    test_data = windows # Für die Anomalieerkennung nehmen wir den gesamten Datensatz

    print(f"Trainingsdaten: {len(train_data)} Fenster")
    print(f"Testdaten: {len(test_data)} Fenster")

    # 4. Modell initialisieren und trainieren
    model = PhysicsAutoencoder(window_size=window_size, latent_dim=8)
    model.compile(optimizer='adam', loss='mse')

    print("Starte Training (30 Epochen)...")
    history = model.fit(
        train_data, train_data, 
        epochs=30, 
        batch_size=128, 
        validation_split=0.1,
        verbose=1
    )

    # 5. Anomalieerkennung
    print("Berechne Rekonstruktionsfehler...")
    reconstructions = model.predict(test_data)
    # MAE pro Fenster berechnen
    mae_loss = np.mean(np.abs(reconstructions - test_data), axis=1)

    # 6. Plotting
    plt.figure(figsize=(15, 6))
    
    # Zeitachse für die Fenster (Mitte des Fensters)
    t = np.arange(len(mae_loss)) / 1000 # In Perioden umrechnen
    
    plt.plot(t, mae_loss, label='Reconstruction Loss (MAE)', color='#1f77b4', alpha=0.8)
    
    # Anomaly Threshold (z.B. das 99. Perzentil der normalen Daten)
    threshold = np.percentile(mae_loss[:split_idx], 99.5)
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Anomaly Threshold ({threshold:.3f})')
    
    # Sabotage-Bereich markieren (70 bis 75)
    plt.axvspan(70, 75, color='red', alpha=0.1, label='Sabotage Area (Ground Truth)')
    
    plt.title('Oracle Anomaly Detection: Reconstruction Loss over Time')
    plt.xlabel('Zeit (Perioden)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Plot speichern
    os.makedirs('data', exist_ok=True)
    plt.savefig('anomaly_detection_plot.png')
    print("Plot gespeichert als 'anomaly_detection_plot.png'")
    
    # Zusätzliche Info für den User
    print(f"Empfohlener Threshold: {threshold:.4f}")

if __name__ == "__main__":
    main()
