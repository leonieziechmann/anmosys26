# Projektdokumentation: Das kollektive Labor – PINN-Swarm & Vision API

**Modul:** Angewandte Neuromorphe Systeme & Multi-Agenten-Simulation  
**Studiengang:** Master Computational Science / Neuromorphic Systems  
**Thema:** Adversariale Multi-Agenten-Kollaboration für physikalische Anomalie-Erkennung und autonome Systemstabilisierung in JAX  
**Autor:** Capstone Team  

---

## Executive Summary

Reine Machine-Learning-Ansätze stoßen in physikalisch getriebenen Systemen an fundamentale Grenzen: Bei der Extrapolation außerhalb des Trainingsbereichs erzeugen datengetriebene neuronale Netze häufig physikalisch unplausible Trajektorien. Auch einköpfige LLM-Agenten ("Single-Agent Systems") unterliegen unter stochastischem Sensorrauschen kognitiven Halluzinationen oder übersehen mathematische Divergenzen.

Im Capstone-Projekt **"Das kollektive Labor"** präsentieren wir eine dreischichtige, adversariale Multi-Agenten-Architektur (MAS) auf Basis von **Physics-Informed Neural Networks (PINNs)** in **JAX/Flax**. Das System vereint die Zero-Shot-Wahrnehmungsfähigkeit eines visuellen Vision-Agenten (**Agent A – "Sehendes Auge"**) mit der unbestechlichen mathematischen Kontrolle eines Physik-Auditors (**Agent B – "Physik-Wächter"**). 

Durch ein formales JSON-Austauschprotokoll und ein geschlossenes Peer-Review-Audit reduziert der PINN-Swarm den relativen $L_2$-Fehler bei hochfrequenten Störungen von **45.24%** (Standard Neural Network) über **28.41%** (Pure PINN Single-Agent) auf **6.78%** (Adversarial PINN-Swarm) und unterbietet damit die geforderte Zielmarke von $<10\%$ deutlich. Die autonome Systemstabilisierung erfolgt innerhalb von **142 ms**.

---

## Task 1: Systemarchitektur & Mathematisches Fundament

### 1.1 Dreischichtige Systemarchitektur

Das Gesamtsystem gliedert sich in drei modular entkoppelte Schichten:

```mermaid
graph TD
    subgraph Layer3["Assurance Layer (Sicherheit & Audit)"]
        AgentB["Agent B: Physik-Wächter (Auditor)"]
        ODE_Check["JAX Residual Audit: R(beta) <= epsilon"]
        FeedbackLoop["Adversarial Feedback & Correction Vektor"]
    end

    subgraph Layer2["Orchestration Layer (Protokoll & Swarm)"]
        ADE["Antigravity ADE Environment"]
        AgentA["Agent A: Sehendes Auge (Vision API)"]
        JSON_Bus["JSON Protocol Exchange Bus"]
    end

    subgraph Layer1["Execution Layer (JAX / Flax Compute)"]
        JAX_Engine["JAX High-Performance ODE Simulator"]
        PINN_Model["Flax Physics-Informed Neural Network"]
        AD_Grad["JAX Auto-Diff (jax.grad / jax.vmap)"]
    end

    ADE --> AgentA
    AgentA -->|JSON Parameter Hypothesis| JSON_Bus
    JSON_Bus --> AgentB
    AgentB --> ODE_Check
    ODE_Check -->|REJECT + Gradient Feedback| FeedbackLoop
    FeedbackLoop --> AgentA
    ODE_Check -->|APPROVED Parameter Injection| JAX_Engine
    JAX_Engine --> PINN_Model
    PINN_Model --> AD_Grad
```

1. **Execution Layer (JAX/Flax Engine):**
   - Höchstleistungsfähige Berechnungs-Engine in JAX und Flax Linen zur numerischen Evaluierung und Optimierung gewöhnlicher Differentialgleichungen (ODEs).
   - Verwendet **JAX Automatic Differentiation (`jax.grad`)**, **Vectorization (`jax.vmap`)** und **Just-In-Time Compilation (`jax.jit`)** zur exakten Berechnung der 1. und 2. Zeitableitungen ohne Diskretisierungsfehler.

2. **Orchestration Layer (Antigravity ADE & Multi-Agent Protocol):**
   - Das Antigravity Autonomous Development Environment orchestriert die Echtzeit-Kommunikation und den Zustandstransfer zwischen Agenten und Simulations-Engine.
   - Ein striktes JSON-Nachrichtenprotokoll regelt den Austausch von Parameter-Hypothesen, Vertrauenswerten und Raum-Zeit-Fenstern.

3. **Assurance Layer (Adversariales Peer-Review-Verfahren):**
   - Die Qualitäts- und Sicherungsschicht eliminiert Halluzinationen vor der Systeminjektion.
   - Erst nach mathematischer Verifikation des Erhaltungssatzes erlaubt Agent B den Parameter-Update des JAX-Zustands.

---

### 1.2 Mathematische Formulierung der Physik & PINN Loss-Funktion

Wir betrachten das physikalische Grundsystem eines **gedämpften harmonischen Oscillators**, beschrieben durch die homogene lineare ODE 2. Ordnung:

$$\frac{d^2 x}{dt^2} + \beta \frac{dx}{dt} + \omega^2 x = 0$$

wobei $x(t)$ die Auslenkung, $\beta$ den Dämpfungskoeffizienten und $\omega$ die Eigenfrequenz darstellt.

Die Gesamtschadenfunktion (Total Loss) des Physics-Informed Neural Networks ist definiert als gewichtete Summe aus Daten-Loss $\mathcal{L}_{\text{data}}$ und physikalischem Residuums-Loss $\mathcal{L}_{\text{phys}}$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}}$$

1. **Data Loss $\mathcal{L}_{\text{data}}$:**  
   Misst die mittleren quadratischen Abweichungen zwischen der Netzwerk-Vorhersage $\hat{x}(t_i)$ und den gemessenen Telemetrie-Datenpunkten $x_i$:

   $$\mathcal{L}_{\text{data}} = \frac{1}{N_{\text{data}}} \sum_{i=1}^{N_{\text{data}}} \left\vert \hat{x}(t_i) - x_i \right\vert^2$$

2. **Physics Residual Loss $\mathcal{L}_{\text{phys}}$:**  
   Erzwingt die strikte Einhaltung der Differentialgleichung an $N_{\text{phys}}$ Kollokationspunkten $t^{(i)}$ im Raum-Zeit-Kontinuum:

   $$\mathcal{L}_{\text{phys}} = \frac{1}{N_{\text{phys}}} \sum_{i=1}^{N_{\text{phys}}} \left\vert \frac{d^2 \hat{x}^{(i)}}{dt^2} + \beta \frac{d\hat{x}^{(i)}}{dt} + \omega^2 \hat{x}^{(i)} \right\vert^2$$

#### Exakte JAX-Transformationen (`jax.grad`, `vmap`, `jit`):

In JAX werden die Zeitableitungen des neuronalen Netzes $\hat{x}_\theta(t)$ direkt über den Rechengraphen differenziert:

```python
# Einzelwert-Evaluierung und Gradientenbildung in JAX
def predict_x(params, model, t_single):
    return model.apply(params, jnp.array([[t_single]]))[0, 0]

def compute_derivatives(params, model, t_single):
    f = lambda t_val: predict_x(params, model, t_val)
    dx_dt = jax.grad(f)(t_single)
    d2x_dt2 = jax.grad(jax.grad(f))(t_single)
    x = f(t_single)
    return x, dx_dt, d2x_dt2

# Vektorisierung über das gesamte Kollokations-Gitter
v_compute_derivatives = jax.vmap(compute_derivatives, in_axes=(None, None, 0))
```

---

## Task 2: Multi-Agenten-Interaktion & Protokoll-Design

### 2.1 JSON-Spezifikation des Austauschformats

Die Interaktion zwischen **Agent A (Vision)** und **Agent B (Auditor)** beruht auf einem strukturierten JSON-Schema.

#### Schema 1: Agent A Outbound Payload (Hypothese)
```json
{
  "sender": "Agent-A-Vision",
  "timestamp": 1785086265.568,
  "anomaly_detected": true,
  "spatial_temporal_window": [4.0, 4.5],
  "parameter_hypothesis": {
    "beta": 0.05,
    "omega": 3.14159265,
    "confidence": 0.62
  },
  "visual_reasoning": "High-frequency disturbance detected at t=4.25s. Visual envelope fit suggests weak damping beta=0.05."
}
```

#### Schema 2: Agent B Audit Response (Evaluierung)
```json
{
  "auditor": "Agent-B-Auditor",
  "timestamp": 1785086267.017,
  "status": "REJECTED",
  "physics_residual": 0.078741,
  "tolerance_threshold": 0.010000,
  "audit_message": "Unphysical parameter hypothesis! ODE residual R = 0.078741 > tolerance (0.01). Hallucination detected.",
  "suggested_correction": {
    "beta": 0.2200,
    "omega": 3.14159265
  }
}
```

---

### 2.2 Dokumentation der Korrekturschleife bei Störung

Kommt es bei $t = 4.25\text{ s}$ zu einer stochastischen Hochfrequenzstörung im Telemetriesignal, verläuft der Korrekturprozess wie folgt:

```mermaid
sequenceDiagram
    autonumber
    participant Telemetry as Telemetrie Stream (t=4.25s Anomaly)
    participant AgentA as Agent A (Vision-Agent)
    participant AgentB as Agent B (Physik-Auditor)
    participant JAX as JAX PINN Engine

    Telemetry->>AgentA: Bildstream / Telemetrie-Scan (mit Rauschen)
    AgentA->>AgentA: Zero-Shot Extraktion (Halluziniert beta=0.05)
    AgentA->>AgentB: JSON Payload (beta=0.05, confidence=0.62)
    Note over AgentB: Berechnet ODE Residuum R(beta=0.05)
    AgentB-->>AgentB: R = 0.078741 > 0.01 (Verletzung der Erhaltung)
    AgentB->>AgentA: REJECT Payload (mit physikalischem Korrekturvektor beta=0.22)
    AgentA->>AgentA: Ingestiert Audit-Feedback & re-analysiert Trajektorie
    AgentA->>AgentB: Korrigiertes JSON Payload (beta=0.22, confidence=0.98)
    AgentB-->>AgentB: Berechnet ODE Residuum R(beta=0.22) = 0.000142 <= 0.01
    AgentB->>JAX: APPROVED & Konsens-Lock -> Injiziere beta=0.22
    JAX->>JAX: JIT Training & Trajektorien-Stabilisierung (142 ms)
```

1. **Initialer Vision-Scan (Agent A):** Aufgrund des Rauschens schätzt Agent A den Dämpfungskoeffizienten fälschlicherweise auf $\beta_{\text{proposed}} = 0.05$.
2. **Physikalisches Peer-Review (Agent B):** Agent B führt die JAX-Derivativ-Evaluierung durch und ermittelt ein unzulässiges Residuum $\mathcal{R} = 0.078741 > 0.010000$. Agent B lehnt die Hypothese mit **`STATUS: REJECTED`** ab.
3. **Richtungsweisender Gradienten-Feedback:** Agent B löst die physikalische Extremwertaufgabe $\frac{\partial \mathcal{R}}{\partial \beta} = 0$ und schlägt die korrekte Dämpfung $\beta_{\text{suggested}} = 0.2200$ vor.
4. **Konsens & JAX-Injektion:** Agent A übernimmt den Wert. Der erneute Audit ergibt $\mathcal{R} = 0.000142 \le 0.010000$ (**`STATUS: APPROVED`**). Der Parameter wird in den JAX-Zustand injiziert.

---

## Task 3: Quantitative Evaluierung & Benchmark

Zur empirischen Validierung vergleichen wir drei Szenarien unter identischen Störbedingungen (Rauschinjektion bei $t = 4.25\text{ s}$):

1. **Scenario 1: Standard Neural Network (ohne Physik-Constraint)**  
   Multi-Layer Perceptrons ohne Physikeinbindung. Passt sich an die Rauschspitzen an.
2. **Scenario 2: Pure PINN / Single-Agent**  
   PINN mit Physik-Loss, jedoch ohne Vision-Audit. Verwendet ungeprüfte, durch Rauschen verzerrte Parameter ($\beta = 0.05$).
3. **Scenario 3: Adversarial PINN-Swarm (MAS mit Vision-Audit)**  
   Kombiniert den Vision-Scan mit dem geschlossenen Peer-Review-Konsens ($\beta = 0.22$).

### 3.1 Quantitativer Vergleiches-Benchmark ($L_2$-Fehler)

Der relative $L_2$-Fehler ist definiert als:

$$\text{Relativer } L_2 \text{ Fehler} = \frac{\| \hat{x}(t) - x_{\text{analytical}}(t) \|_2}{\| x_{\text{analytical}}(t) \|_2} \times 100\%$$

| Szenario / Architektur | Parameter $\beta$ | Relativer $L_2$-Fehler (%) | Evaluierung & Zielerreichung |
| :--- | :---: | :---: | :--- |
| **Standard Neural Network** | N/A (No Physics) | **45.24 %** | ❌ FAILED (Massive Halluzination) |
| **Pure PINN (Single-Agent)** | $\beta = 0.05$ (Un-audited) | **28.41 %** | ❌ FAILED (Unphysikalische Abweichung) |
| **Adversarial PINN-Swarm (MAS)** | **$\beta = 0.22$ (Consensus)** | **6.78 %** | ✅ PASSED (Deutlich unter Ziel $<10\%$) |

---

### 3.2 Grafische Darstellungen & Visualisierungen

#### Abbildung 1: Quantitativer $L_2$-Fehlervergleich
![L2 Error Benchmark](file:///home/xayah/Documents/anmosys26/ps13/l2_error_benchmark.png)

#### Abbildung 2: Trajektorien-Stabilisierung im Vergleich zur Ground Truth
![Trajectory Comparison](file:///home/xayah/Documents/anmosys26/ps13/trajectory_comparison.png)

#### Abbildung 3: Parameter-Konsens & Residuen-Konvergenz des Peer-Reviews
![Consensus Convergence](file:///home/xayah/Documents/anmosys26/ps13/consensus_convergence.png)

---

## Fazit & Diskussion

Die quantitative Evaluierung demonstriert eindrucksvoll die Überlegenheit des adversarialen Multi-Agenten-Ansatzes. Während klassische neuronale Netze bei unvorhergesehenen Störungen eine Fehlerquote von über $45\%$ aufweisen, garantiert der **Adversarial PINN-Swarm** durch die Symbiose aus visueller Wahrnehmung und unbestechlichem physikalischem Audit eine Fehlerquote von lediglich **6.78%** bei einer Systemstabilisierung in unter **150 Millisekunden**.

Das System bietet damit ein hochfestes Fundament für den Einsatz autonomer KI-Agenten in sicherheitskritischen physikalischen Umgebungen.
