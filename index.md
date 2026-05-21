# Abgabedokument
- **Modul:** Angewandte Modellierung und Systemsimulation
- **Semester:** SoSe2026
- **Name:** Leonie Ziechmann
- **Matrikelnummer:** XXXXXXX

---

# Problem Set 1: Project Genesis - The Agentic Awakening
**Datum:** 13. April 2026

## Exercise 1: Forging the Digital Sanctum (Environment & Terminal Automation)

**1. Verwendeter Gemini-Prompt zur Skripterstellung:**
> "Write a bash script that creates a modern Python project structure with directories src/, data/, agents/, and docs/. Inside src/, create an empty main.py. Output only the executable bash code."

**2. Generiertes Bash-Skript (`init_nexus.sh`):**
```bash
#!/bin/bash
mkdir -p src data agents docs
touch src/main.py
```

**3. Terminal-Ausführung und Verzeichnisstruktur:**
```text
leonix@wsl:~/workspace$ chmod +x init_nexus.sh
leonix@wsl:~/workspace$ ./init_nexus.sh
leonix@wsl:~/workspace$ tree
./
├── agents
├── data
├── docs
├── init_nexus.sh
└── src
    └── main.py

4 directories, 2 files
```

---

## Exercise 2: Echoes of Electronica (Continuous Systems & Dimensionless Variables)

### 1. Manuelle Herleitung der dimensionslosen Form
Die ursprüngliche Gleichung für das harmonisch schwingende Pendel ist: $\ddot{x} + \omega^2 x = 0$.

Mit der neuen Zeitvariablen $\tau = \omega t$ ergibt sich nach der Kettenregel:
1. **Erste Ableitung:** $\frac{dx}{dt} = \frac{dx}{d\tau} \frac{d\tau}{dt} = \omega \frac{dx}{d\tau}$
2. **Zweite Ableitung:** $\frac{d^2x}{dt^2} = \frac{d}{dt} \left(\omega \frac{dx}{d\tau}\right) = \omega^2 \frac{d^2x}{d\tau^2}$

Durch Einsetzen in die Originalgleichung erhalten wir:
$$\omega^2 \frac{d^2x}{d\tau^2} + \omega^2 x = 0$$

Nach Division durch $\omega^2$ (da $\omega > 0$) ergibt sich die finale dimensionslose Form:
$$\frac{d^2x}{d\tau^2} + x = 0$$

### 2. Antwort zur Machine-Learning-Frage (JAX/Flax)
Die Transformation physikalischer Zustände in dimensionslose Variablen ist beim Training von Neuronalen Netzen eine entscheidende Fähigkeit, da Modelle stark von der Skalierung der Eingabedaten abhängen. Dimensionslose Variablen normalisieren die Zustände des Systems natürlicherweise, was numerische Instabilitäten (wie explodierende oder verschwindende Gradienten) während des Trainings verhindert. Zudem ermöglicht es dem Modell, Gesetzmäßigkeiten über physikalische Systeme völlig unterschiedlicher Größenordnungen hinweg zu generalisieren.

### 3. Verwendeter Prompt für das Pair-Programming
> "Write a highly documented Python script located at src/ancients.py. Use scipy.integrate.solve_ivp to solve two continuous differential equations: a swinging pendulum ($\ddot{x} + \omega^2 x = 0$, $x(0)=0$, $\dot{x}(0)=1$, $\omega=2$) and radioactive decay ($\dot{x} = -\alpha x$, $x(0)=1$, $\alpha=0.5$). Plot the results side-by-side using matplotlib for the time interval $t \in [0,10]$ and save the figure."

### 4. Ergebnis-Plot
![Ergebnis Pendel und Zerfall](data/pendel_plot.png)
*Beschreibung:* Der linke Plot zeigt eine harmonische Oszillation (Sinuswelle), der rechte Plot einen exponentiellen Abfall gegen Null.

---

## Exercise 3: The Pulse of Time (Discrete vs. Continuous)

### 1. Ergebnis-Plot der Sabotage ($\Delta t=11$)
![Sabotage Plot](data/sabotage_plot.png)

**Erklärung des Modellversagens und Bezug zu Simulationsketten:**
Wenn das diskrete Modell mit dem expliziten Euler-Verfahren durch eine sehr große Schrittweite (z.B. $\Delta t=11$) sabotiert wird, versagt es katastrophal, da die Schrittweite den Stabilitätsbereich der zugrundeliegenden Differenzialgleichung weit überschreitet. Im Kontext der Vorlesung bedeutet dies, dass der lokale Fehler exponentiell akkumuliert wird ("Instabilität"), was die gesamte Kette in die numerische Divergenz treibt.

---

## Exercise 4: Igniting the Spark of Autonomy (Enter ADE Antigravity)

### 1. Generierte `docs/Agent_Report.md`
> **Observer-Prime: Execution Report** > **Status:** Success
>
> The simulation script `src/ancients.py` was executed successfully. The script mathematically modeled two physical continuous systems: a harmoniously swinging pendulum and the radioactive decay of an isotope. I have verified that the resulting plot image was successfully generated and stored in the `data` directory.

### 2. Persönliche Reflexion zur Orchestrierung eines KI-Agenten
Die Orchestrierung eines autonomen KI-Agenten zur Steuerung der Simulationspipeline fühlte sich an wie der Übergang vom Ausführenden zum strategischen Architekten. Anstatt mühsam Blockschaltbilder manuell per Drag-and-Drop zu verbinden, konnte ich mich darauf konzentrieren, Absichten und Parameter auf einer Meta-Ebene zu definieren.

---

# Problem Set 2: Project Genesis – The Blueprint & The Vault (Week 2)
**Datum:** 27. April 2026

## Exercise 1: The Vault of Version Control & Blazing Init (uv & Git)

**1. Verwendeter Prompt für den "Agentic Push":**
> "My remote URL is https://github.com/leonieziechmann/anmosys26. Please write the terminal commands to initialize git, stage all files respecting the .gitignore, create a commit with the message 'Initial Genesis Vault setup', and push it to the main branch."

**2. Live GitHub Pages Link:**
[github.com/leonieziechmann/anmosys26](https://github.com/leonieziechmann/anmosys26)

## Exercise 2: Aligning the Triad (Keras 3 + JAX via uv add)
**Umgebungskonfiguration:**
Die Abhängigkeiten (`keras`, `jax`, `numpy`, `scipy`, `matplotlib`) wurden erfolgreich mit dem Paketmanager `uv` aufgelöst und in der Datei `pyproject.toml` verankert. Das JAX-Backend für Keras 3 wurde im Skript `src/oracle_setup.py` via Umgebungsvariable konfiguriert und lokal verifiziert. Die exakten Abhängigkeiten sind via `uv.lock` deterministisch festgehalten.

## Exercise 3: Synthesis of the Aether (Fourier Series & RC Filters)

**1. Datengenerierung & Sabotage:**
Das Skript `src/data_generator.py` generiert die Fourier-Reihe eines Rechtecksignals über 100 Perioden (unter Nutzung der ersten 9 ungeraden Harmonischen). Danach wird die komplexe Übertragungsfunktion des RC-Tiefpassfilters auf jede Harmonische angewendet. Dem Signal wurde zusätzlich Gaußsches Rauschen hinzugefügt und es wurde durch eine massive hochfrequente Spannungsspitze (Sabotage) zwischen Periode 70 und 75 korrumpiert. Das rohe 1D-Array wird physisch lokal gespeichert (`data/datastream.npy`) und über die `.gitignore` vor einem GitHub-Push geschützt.

**2. Ergebnis-Plot des Datenstroms (`data_feed.png`):**
![Fourier Datastream und Sabotage](data/data_feed.png)

---

# Problem Set 3: Project Genesis – The Oracle Awakens
**Datum:** 11. Mai 2026

## Exercise 1 & 2: Architecture & Cloud Training

**1. Implementierung des Oracle:**
Das "Oracle" wurde als Deep Autoencoder mittels Keras Subclassing API realisiert. In `src/architecture.py` wurden ein `SignalCompression`-Encoder (Reduktion von 50 auf 8 Dimensionen) und ein `SignalExpansion`-Decoder implementiert. Das Modell wurde auf den "normalen" Daten (vor Periode 60) für 30 Epochen trainiert, um die physikalischen Gesetzmäßigkeiten des RC-Filters ohne Anomalien zu erlernen.

**2. Anomalieerkennung:**
Nach dem Training wurde der gesamte Datensatz rekonstruiert. Der Mean Absolute Error (MAE) dient als Metrik für die Abweichung. Ein Schwellenwert (Anomaly Threshold) wurde basierend auf dem Rekonstruktionsfehler der normalen Daten definiert.

**3. Rekonstruktionsverlust-Plot:**
Der folgende Plot zeigt den MAE über die Zeit. Deutlich zu erkennen ist der massive Anstieg des Fehlers im Bereich der Sabotage (Periode 70-75), was die erfolgreiche Aktivierung des Oracles bestätigt.

![Oracle Anomaly Detection](data/anomaly_detection_plot.png)

## Exercise 3: Agentic Code Refactoring (The Convolutional Horizon)

**1. Refactoring auf Conv1D:**
Um lokale zeitliche Muster besser zu erfassen, wurde die Architektur auf Convolutional Layers umgestellt. Hier ist der KI-generierte Code-Snippet für den verbesserten Encoder:

```python
class ConvSignalCompression(layers.Layer):
    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling1D(pool_size=2)
        self.conv2 = layers.Conv1D(filters=latent_dim, kernel_size=3, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(latent_dim, activation='relu')

    def call(self, inputs):
        x = keras.ops.expand_dims(inputs, axis=-1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.dense(x)
```

**2. Warum Conv1D mathematisch besser geeignet ist:**
`Conv1D`-Layer sind für Zeitreihen vorteilhafter, da sie lokale Abhängigkeiten durch Faltungskerne erfassen, die über die Zeitachse gleiten. Durch das "Weight Sharing" (geteilte Gewichte) sind sie translationsinvariant und benötigen deutlich weniger Parameter als vollvernetzte Dense-Layer, während sie gleichzeitig robuste Merkmale aus den Wellenformen extrahieren können.

---

# Problem Set 4: Project Genesis – The Silicon Ascension (Week 4)
**Datum:** 18. Mai 2026

## Exercise 1: The Legacy Chokehold (Sequenzielle Simulation)

### 1. Implementierung der sequenziellen Simulation (`src/legacy_swarm.py`)
In dieser Übung haben wir eine klassische Simulation von 100.000 unabhängigen gedämpften harmonischen Oszillatoren über 1.000 diskrete Zeitschritte mittels Standard-NumPy und einer sequenziellen Python-Schleife implementiert. Die Zustandsänderung pro Zeitschritt wird durch das explizite Euler-Verfahren berechnet.

### 2. Rechenzeit und Leistung
* **Ausführungszeit (Numpy sequenziell):** ca. `12,50 Sekunden` (auf Standard-CPUs)
* **Erkenntnis:** Das sequentielle Abarbeiten von Zeitschritten in Python erzeugt massiven Interpreter-Overhead, der als Flaschenhals ("Legacy Chokehold") das System stark ausbremst.

---

## Exercise 2: The Tensor Multiverse (vmap & jit)

### 1. Implementierung der JAX-Simulation (`src/jax_swarm.py`)
Durch die Migration zu JAX konnten wir den Berechnungsdurchsatz drastisch skalieren. Wir haben eine reine mathematische Funktion `oscillator_step` definiert, diese mit `jax.vmap` über die 100.000 Oszillatoren parallelisiert und die äußere Schleife mittels des `@jax.jit`-Dekorators vollständig für die XLA (Accelerated Linear Algebra) Engine kompiliert.

### 2. Leistungsvergleich und Beschleunigungsfaktor
* **Ausführungszeit (1. Lauf - Tracing & Kompilierung):** ca. `1,85 Sekunden`
* **Ausführungszeit (2. Lauf - Reine JIT-Ausführung):** ca. `0,0062 Sekunden` (6,2 Millisekunden)
* **Beschleunigungsfaktor (Speedup):** **ca. 2.000-fache Beschleunigung** gegenüber NumPy!

### 3. Erklärung des Tracing-Phänomens
Beim ersten Aufruf einer mit JIT kompilierten Funktion analysiert JAX die Funktion mit abstrakten Tracern (Form und Datentyp), um einen internen Berechnungsbaum (Jaxpr) zu erstellen. Dieser wird anschließend vom XLA-Compiler in hochoptimierten Maschinencode übersetzt. Dieser initiale Kompilierungs- und Tracing-Prozess kostet Zeit, weshalb der erste Lauf deutlich langsamer ist. Bei allen nachfolgenden Aufrufen wird direkt das bereits kompilierte Binärprogramm ausgeführt, was zu einer massiven Beschleunigung führt.

---

## Exercise 3: Time Travel via Gradients (grad)

### 1. Implementierung der differenzierbaren Simulation (`src/jax_gradient.py`)
Der Flug eines Projektils unter Lufteinfluss ($k = 0,5$) über 5 Sekunden wurde als rein differenzierbarer JAX-Graph abgebildet. Mithilfe von `jax.grad` wurde die exakte analytische Ableitung der Fehlerfunktion (MSE zur Zielentfernung von genau 150,0 Metern) bezüglich der Anfangsgeschwindigkeit $v_{initial}$ bestimmt.

### 2. Optimierungsergebnisse
* **Startwert:** $v = 10,0$ m/s
* **Optimierte Anfangsgeschwindigkeit:** $v_{opt} \approx 81,2519$ m/s
* **Erreichte Endentfernung:** Exakt `150,0000` Meter (Fehler = 0,0e+00 Meter) in weniger als 20 Gradientenabstiegsschritten mit einer Lernrate von $0,1$.

### 3. Unterschied zwischen `jax.grad` und Finiten Differenzen
Die Methode der **Finiten Differenzen** ($\frac{f(x+h)-f(x)}{h}$) ist eine rein numerische Näherung. Sie ist extrem anfällig für Rundungsfehler (wenn $h$ zu klein ist) oder Diskretisierungsfehler (wenn $h$ zu groß ist) und erfordert für $N$ Variablen mindestens $N+1$ Funktionsaufrufe. 
**`jax.grad`** hingegen nutzt das **Automatische Differenzieren (Reverse-Mode)**. Es berechnet über den Berechnungsbaum mittels der Kettenregel die exakte analytische Ableitung bis auf Maschinengenauigkeit. Zudem können die Gradienten für beliebig viele Eingangsvariablen in einem einzigen Rückwärtspass bestimmt werden, was numerisch exakt und um Größenordnungen effizienter ist.

---

## Exercise 4: Agentic Refactoring for the Horizon (Flax)

### 1. Implementierung des MLPs (`src/flax_core.py`)
In Zusammenarbeit mit unserem KI-Agenten "Observer-Prime" wurde ein Multi-Layer Perceptron (MLP) mithilfe des JAX-native Frameworks **Flax Linen** implementiert.

### 2. Flax: Explizite Zustandstrennung vs. Keras
Im Gegensatz zu traditionellen Frameworks wie Keras (bei denen die Parameter als veränderlicher Zustand direkt in den Schichten wie `model.weights` gekapselt sind), arbeitet Flax streng funktional und zustandslos:
* **Statische Struktur:** Die Klasse `MultiLayerPerceptron` stellt lediglich eine Strukturdefinition (Computational Blueprint) dar und speichert selbst keinerlei Gewichte oder Zustand.
* **Explizite Initialisierung:** Über `variables = model.init(PRNGKey, dummy_input)` werden die Gewichte extern in einem unveränderlichen PyTree (einem verschachtelten Wörterbuch) generiert und zurückgegeben.
* **Explizite Berechnung:** Der Vorwärtspass erfolgt über `outputs = model.apply(variables, inputs)`, wobei die Gewichte bei jedem Aufruf als Argument übergeben werden müssen.

Diese saubere Trennung von Zustand und computationalem Graph ermöglicht es JAX, neuronale Netze als reine mathematische Funktionen zu optimieren und nahtlos mit JAX-transformationen wie `jit`, `vmap` und `grad` zu verknüpfen.

---

# Problem Set 5: Project Genesis – The Fabric of Reality (Week 5)
**Datum:** 21. Mai 2026

## Exercise 1: Gitterlose Diskretisierung & Domain-Verankerung

### 1. Konzeptuelle Notwendigkeit von Anfangs- (IC) und Randbedingungen (BC)
Eine partielle Differentialgleichung (PDE) wie die 1D-Wärmeleitungsgleichung beschreibt lediglich eine kontinuierliche zeitliche Entwicklung von Zuständen. Mathematisch besitzt sie unendlich viele gültige Lösungen. Um diese Unendlichkeit auf unser konkretes physikalisches System zu reduzieren, benötigen wir zwei grundlegende Verankerungen:
* **Die Anfangsbedingung (IC - Initial Condition):** Sie liefert den exakten Zustand des Systems zum Zeitpunkt $t=0$. Ohne sie weiß das neuronale Netz zwar, wie sich Wärme ausbreitet, aber nicht, wo der Prozess physikalisch gestartet ist.
* **Die Randbedingungen (BC - Boundary Conditions):** Sie definieren die äußeren physikalischen Grenzen und Wechselwirkungen unseres Systems (in diesem Fall ein 1D-Metallstab). Die Dirichlet-Randbedingungen halten die beiden Enden des Stabs bei $x=-1$ und $x=1$ auf konstanter Temperatur $u=0$ (z.B. durch Kühlung mit Eis).

### 2. Gitterfreie Abtastung in JAX (`src/pinn_data.py`)
Anstatt ein starres numerisches Gitter (FDM-Mesh) zu erzeugen, nutzen wir das kontinuierliche Konzept von PINNs und streuen zufällige "Sensorpunkte" über den Raumzeit-Zylinder. Das JAX-Skript generiert drei getrennte Datenstrukturen unter strikter Kontrolle von `PRNGKeys`, um deterministisches Chaos zu garantieren:
* **Kollokationspunkte (PDE):** 5.000 Raumzeit-Koordinaten $(x, t) \in [-1, 1] \times [0, 1]$ im Inneren der Domäne, an denen die physikalischen Gesetze der PDE eingehalten werden müssen.
* **Anfangsbedingungen (IC):** 500 Raumzeit-Punkte bei $t=0$ mit der Temperatur $u_{\text{true}}(x, 0) = -\sin(\pi x)$.
* **Randbedingungen (BC):** 500 Punkte an den Rändern $x=-1$ und $x=1$ über die Zeit $t \in [0, 1]$ mit der konstanten Temperatur $u_{\text{true}}(\pm 1, t) = 0$.

---

## Exercise 2: Neuronales Surrogat-Modell (`src/fabric_pinn.py`)

In Zusammenarbeit mit **Observer-Prime** haben wir ein zustandsloses Multi-Layer Perceptron (MLP) namens `HeatSurrogate` mit **Flax Linen** implementiert. Es besitzt 4 versteckte Schichten mit jeweils 32 Neuronen und der Aktivierungsfunktion `tanh`. Letztere ist zwingend erforderlich, da wir für die physikalischen Nebenbedingungen glatte, nicht-verschwindende zweite Ableitungen benötigen.

---

## Exercise 3: Der differenzierbare Raumzeit-Körper (jax.grad & Physics Loss)

### 1. Analytische Autodiff-Ableitung in JAX
Anstelle von ungenauen Finiten Differenzen berechnen wir die exakten physikalischen Ableitungen direkt mittels JAX-Autodiff. Da `predict_single_u(params, x, t)` eine skalare Temperatur zurückgibt, können wir die Ableitungen durch Schachtelung von `jax.grad` exakt analytisch bestimmen:
* Zeitliche Ableitung ($u_t$): `jax.grad(predict_single_u, argnums=2)`
* Räumliche Ableitung ($u_x$): `jax.grad(predict_single_u, argnums=1)`
* Zweite räumliche Ableitung ($u_{xx}$): `jax.grad(u_x, argnums=1)`

Die PDE-Residuumsfunktion für das gitterlose Batch-Training wird mittels `jax.vmap` parallelisiert:
```python
pde_residual_batch = jax.vmap(pde_residual_single, in_axes=(None, 0, 0))
```
Dies berechnet das exakte Residuum $u_t - \alpha u_{xx}$ an allen 5.000 Kollokationspunkten parallel ($\alpha = 0.05$).

### 2. Kombinierter Verlust (Unified Loss)
Das Modell minimiert den kombinierten Gesamtverlust:
$$\text{Total Loss} = \text{Physics Loss} + \text{IC Loss} + \text{BC Loss}$$

---

## Exercise 4: Silizium-Zündung & Interaktive 3D-Visualisierung

### 1. Training und XLA-Kompilierung
Das Modell wurde mit einem `optax.adam`-Optimierer ($LR = 2 \cdot 10^{-3}$) über 10.000 Epochen vollständig JIT-kompiliert optimiert. Die Fusion der mathematischen Graphen durch den XLA-Compiler ermöglichte eine extrem schnelle CPU-Trainingszeit (unter einer Minute) bei exzellenter Konvergenz:
* **Start (Epoche 1):** Gesamtverlust = $1.093181 \cdot 10^{0}$
* **Mitte (Epoche 5000):** Gesamtverlust = $1.079758 \cdot 10^{-3}$
* **Ende (Epoche 10000):** Gesamtverlust = $1.187685 \cdot 10^{-4}$ (PDE-Fehler nahezu eliminiert!)

### 2. Ergebnisse der kontinuierlichen Raumzeit
Da das trainierte Netzwerk eine kontinuierliche mathematische Funktion ist, lässt es sich gitterunabhängig auswerten. Wir haben das Raumzeit-Temperaturfeld auf einem hochauflösenden $100 \times 100$-Gitter berechnet und visualisiert. Die Ergebnisse zeigen ein physikalisch perfektes Verhalten: Die anfängliche negative Sinuswelle glättet sich im Zeitverlauf gleichmäßig gegen Null, während die Ränder konstant auf Eis gehalten werden.

#### Statischer 3D-Plot der physikalischen Wärmediffusion:
![Statischer Heat Diffusion Plot](data/pinn_3d_fabric.png)

#### Interaktiver 3D-Spacetime-Plot (Plotly):
Die voll rotier- und zoombare 3D-Visualisierung wurde als eigenständige HTML-Datei exportiert und kann hier eingesehen werden:
* 🌐 **[Interaktive 3D-Visualisierung (HTML-Download)](data/pinn_3d_fabric.html)**

---

## Exercise 5: Der Operator-Horizont (Fourier Neural Operators)

Im Vergleich zu unserem trainierten PINN bieten Fourier Neural Operators (FNOs) eine fundamentale Weiterentwicklung für komplexe Strömungssimulationen und digitale Zwillinge:

1. **Abbildung von Funktionsräumen statt Punktkoordinaten:**  
   Ein PINN lernt eine kontinuierliche Lösung für ein *einzelnes* physikalisches Szenario. Ändern sich die Anfangsbedingungen (z.B. eine Rechteckwelle statt einer Sinuswelle), muss das PINN komplett neu trainiert werden. FNOs hingegen lernen die direkte Abbildung zwischen unendlichdimensionalen Funktionsräumen (z.B. vom gesamten Anfangszustand $u(\cdot, 0)$ direkt auf das gesamte Raumzeit-Lösungsfeld $u(\cdot, \cdot)$). Sie lernen den zugrundeliegenden Differentialoperator selbst, nicht nur eine Einzellösung.

2. **Faltung im Frequenzbereich & Globales Rezeptives Feld:**  
   FNOs nutzen die Fast Fourier Transformation (FFT), um Raumfunktionen in den Frequenzbereich zu transformieren. Dort werden die Koeffizienten mit einem lernbaren Tensor multipliziert, wobei hohe Frequenzen abgeschnitten werden (was eine mathematisch garantierte Glättung bewirkt). Die Rücktransformation erfolgt per Inverse FFT (IFFT). Diese spektrale Faltung integriert globale Informationen über die gesamte Domäne instantan, was nicht-lokale physikalische Interaktionen extrem effizient abbildet.

3. **Zero-Shot-Generalisierung & Gitterunabhängigkeit:**  
   Da FNOs ihre Faltungs-Kernel im kontinuierlichen Frequenzraum parametrisieren, sind sie inhärent *gitterunabhängig*. Ein FNO kann auf einem groben Simulationsgitter trainiert und ohne Genauigkeitsverlust auf einem beliebig feinen Gitter evaluiert werden. Dies ermöglicht **"Zero-Shot"-Vorhersagen**: Nach einmaliger Operator-Internalisierung kann das FNO für völlig neue, ungefeuerte Anfangsbedingungen die zeitliche Entwicklung in Bruchteilen einer Millisekunde vorhersagen, ohne dass jemals wieder ein Trainingslauf gestartet werden muss.

