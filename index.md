# Abgabedokument
**Modul:** Angewandte Modellierung und Systemsimulation
**Semester:** SoSe2026
**Datum:** 13. April 2026
**Thema:** Problem Set 1: Project Genesis - The Agentic Awakening
**Name:** Leonie Ziechmann
**Matrikelnummer:** XXXXXXX

---

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
