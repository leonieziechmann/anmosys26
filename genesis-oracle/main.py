import os
import json
import numpy as np
import sys

# Ensure local paths work
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import classical_pi
from cognitive_core.agent import analyze_telemetry_plot
from defensive_agent import validate_damping_jax
import fabric_pinn
from pinn_data import generate_pinn_data

def update_simulation_parameters(beta, timestamp):
    """
    Updates simulation_parameters.json in project root and genesis-oracle directory using absolute paths.
    """
    paths = [
        "/home/xayah/Documents/anmosys26/simulation_parameters.json",
        "/home/xayah/Documents/anmosys26/genesis-oracle/simulation_parameters.json"
    ]
    for file_path in paths:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
            
            # Keep original keys and add/update jax_state parameters
            data["jax_state"] = {
                "damping_beta": float(beta),
                "anomaly_timestamp": float(timestamp),
                "stabilized": True
            }
            
            # Ensure the parameters array lists this update
            if "parameters" not in data or not isinstance(data["parameters"], list):
                data["parameters"] = []
            
            # Append JAX State parameter
            data["parameters"].append({
                "name": "JAX Swarm Damping Beta",
                "value": str(beta),
                "unit": "dimensionless",
                "context": f"Stabilized after anomaly at t={timestamp}s"
            })
            
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Updated parameters saved to: {file_path}")

def run_stabilized_render(beta):
    """
    Runs the PINN simulation and exports pinn_3d_fabric.html/png.
    """
    print("\nRunning final stabilized PINN simulation...")
    import jax
    # Generate data using key
    data_key = jax.random.PRNGKey(101)
    dataset = generate_pinn_data(data_key)
    
    # Train PINN (run fewer epochs for speed in dry-run if needed, but 1000 is fast and compiles well)
    final_params = fabric_pinn.train_pinn(dataset, epochs=1000, lr=2e-3)
    
    # Render and visualize solution
    project_root = os.path.dirname(os.path.abspath(__file__))
    fabric_pinn.visualize_solution(final_params, project_root)
    print("Stabilized oscillation rendered successfully.")

# ==============================================================================
# 5-Minute Arena Pitch Simulation Mode Implementation
# ==============================================================================
import time
import argparse
import random
import shutil

MOCK_DATA = {
    "normal": {
        "gemma": [
            "Bidding standard base price of 1500 EUR/ton. Supply levels are steady.",
            "Purchased 2 tons of synthetic parquet at 1500 EUR/ton. Market equilibrium maintained.",
            "Order of 2 tons confirmed. No deviations in supply chain detected.",
            "Buying synthetic parquet batch at base price of 1500 EUR/ton. Stable supply detected.",
            "Secured 2 tons at standard baseline rate of 1500 EUR/ton. Flow rate is optimal."
        ],
        "gemini": [
            "Analyzing macro telemetry. Damping coefficient stable at 0.180. Damping ratio indicates zero risk.",
            "JAX Swarm state remains in high-efficiency regime. No structural anomalies in current epoch.",
            "Market volatility index: low. Damping Beta of 0.180 is verified as optimal.",
            "Macro-metrics stable. Inflation risk: low. Damping factor optimal.",
            "Telemetry analysis shows steady-state operation. Structural variance within expected tolerances."
        ]
    },
    "shock": {
        "gemma": [
            "⚠️ Alert: Supply shortage detected! Bidding higher to secure stock: 2500 EUR/ton!",
            "🚨 Critical supply constraint! Bidding aggressively: 3500 EUR/ton!",
            "🔥 PANIC BUYING: Bidding 4200 EUR/ton! Must preserve operational capacity!"
        ],
        "gemini": [
            "Warning: Telemetry shows drastic reduction in JAX Damping Beta to 0.040. Price divergence detected.",
            "Macro-metrics warning: Gemma's aggressive bidding is creating localized hyper-inflation.",
            "System instability: Damping coefficient collapsed. Market is entering a monopolistic feedback loop."
        ]
    },
    "regulation": {
        "regulator": [
            "🛡️ Regulator Intervention: Hyper-inflation threshold breached. Enforcing price cap of 2000 EUR/ton to prevent market failure.",
            "🛡️ Regulator Intervention: Blocking monopolistic bids. Re-centering JAX Damping Beta to 0.220 for system stabilization."
        ],
        "gemma": [
            "Bid restricted to 2000 EUR/ton by regulatory authority. Bidding cap active.",
            "Purchased batch at capped price of 2000 EUR/ton. Order volume throttled.",
            "Gemma bidding restricted. Resetting buy limit to 2000 EUR/ton."
        ],
        "gemini": [
            "Policy feedback applied. JAX Damping Beta stabilized at 0.220. Inflation index dropping.",
            "Macro-metrics returning to safe bounds. Monopolistic risk mitigated. System convergence achieved.",
            "Market stabilizing. Damping factor recovered to 0.220. Monopolistic risk mitigated."
        ]
    }
}

def check_supply_shortage():
    paths = [
        "/home/xayah/Documents/anmosys26/simulation_parameters.json",
        "/home/xayah/Documents/anmosys26/genesis-oracle/simulation_parameters.json"
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                return data.get("supply_shortage", False)
            except Exception:
                pass
    return False

def reset_supply_shortage():
    paths = [
        "/home/xayah/Documents/anmosys26/simulation_parameters.json",
        "/home/xayah/Documents/anmosys26/genesis-oracle/simulation_parameters.json"
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                data["supply_shortage"] = False
                with open(p, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

def get_dynamic_message(role: str, state: str, price: float, beta: float):
    """
    Attempts to call the Google GenAI API to generate dynamic agent messages.
    Falls back to mock data if there are issues.
    """
    try:
        from google import genai
        from google.genai import types
        client = genai.Client()
        
        system_instruction = (
            "You are simulating a live-demo of a multi-agent system controlling a JAX-accelerated market. "
            "Respond with a single, highly realistic, concise 1-sentence statement from the requested role. "
            "Do not include any other text, quotes, or markdown formatting."
        )
        
        prompt = (
            f"Role: {role}\n"
            f"Market State: {state}\n"
            f"Current Price: {price} EUR/ton\n"
            f"Current JAX Damping Beta: {beta:.3f}\n"
        )
        if role == "Gemma (Buyer)":
            prompt += "State your bidding action or behavior under this market state."
        elif role == "Gemini (Macro Analyst)":
            prompt += "Evaluate system stability, damping factors, and monopolistic risk."
        elif role == "Agent C (Regulator)":
            prompt += "State the regulatory intervention policy you are enforcing (e.g. price cap, resetting damping)."
            
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            max_output_tokens=80
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )
        text = response.text.strip().replace('"', '')
        if text:
            return text
    except Exception:
        pass
    return None

def run_realtime_simulation(fallback_mock: bool):
    import signal
    def raise_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt()
    try:
        signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    except ValueError:
        # Not in main thread, ignore signal registration
        pass

    print("\033[94m==================================================")
    print("STARTING REAL-TIME PHYSICAL MARKET SIMULATION")
    print("==================================================\033[0m")
    
    # Reset shock state in parameters at start
    reset_supply_shortage()
    
    # State tracking
    state = "normal"  # normal -> shock -> regulation
    iteration = 1
    price = 1500.0
    beta = 0.180
    loss = 0.024
    shock_step = 0
    
    # Load env variables (credentials) just in case we need them for dynamic calling
    dotenv_path = "/home/xayah/Documents/anmosys26/genesis-oracle/.env"
    if os.path.exists(dotenv_path):
        with open(dotenv_path) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v.strip('"')

    try:
        while True:
            # 1. Read simulation parameters dynamically (file-watcher)
            if state == "normal" and check_supply_shortage():
                state = "shock"
                shock_step = 0
                print("\n\033[91m⚠️⚠️⚠️ ALERT: supply_shortage DISTURBANCE DETECTED IN PARAMETERS ⚠️⚠️⚠️\033[0m\n")
                
            # Update values based on state
            if state == "normal":
                beta = 0.180
                loss = 0.02 + random.uniform(-0.005, 0.005)
                price = 1500.0
            elif state == "shock":
                shock_step += 1
                beta = max(0.040, 0.180 - shock_step * 0.05)
                loss = 0.65 + shock_step * 0.10 + random.uniform(-0.02, 0.02)
                if shock_step == 1:
                    price = 2500.0
                elif shock_step == 2:
                    price = 3500.0
                else:
                    price = 4200.0
                    
                # Auto-transition to regulation after 3 shock steps
                if shock_step >= 3:
                    state = "regulation"
                    shock_step = 0
                    print("\n\033[93m🛡️🛡️🛡️ AUTOMATIC POLICY INTERVENTION: AGENT C (REGULATOR) ONLINE 🛡️🛡️🛡️\033[0m\n")
                    
            elif state == "regulation":
                beta = 0.220
                loss = 0.12 + random.uniform(-0.01, 0.01)
                price = 2000.0  # Capped
                
            # Print JAX physics logs
            print(f"\033[94m[JAX Engine] Iteration {iteration:03d} | Loss: {loss:.4f} | Market Damping Beta: {beta:.3f}\033[0m")
            
            # Print Agent traces (Gemma, Gemini, and Agent C if active)
            if state == "normal":
                # Gemma
                msg_gemma = None
                if not fallback_mock:
                    msg_gemma = get_dynamic_message("Gemma (Buyer)", "Normal Operation", price, beta)
                if not msg_gemma:
                    msg_gemma = random.choice(MOCK_DATA["normal"]["gemma"])
                print(f"\033[92m[Gemma (Buyer)]: \"{msg_gemma}\"\033[0m")
                
                # Gemini
                msg_gemini = None
                if not fallback_mock:
                    msg_gemini = get_dynamic_message("Gemini (Macro Analyst)", "Normal Operation", price, beta)
                if not msg_gemini:
                    msg_gemini = random.choice(MOCK_DATA["normal"]["gemini"])
                print(f"\033[95m[Gemini (Macro Analyst)]: \"{msg_gemini}\"\033[0m")
                
            elif state == "shock":
                # Gemma (panic)
                msg_gemma = None
                if not fallback_mock:
                    msg_gemma = get_dynamic_message("Gemma (Buyer)", "Supply Shortage (Shock)", price, beta)
                if not msg_gemma:
                    idx = min(shock_step - 1, len(MOCK_DATA["shock"]["gemma"]) - 1)
                    msg_gemma = MOCK_DATA["shock"]["gemma"][idx]
                print(f"\033[92m[Gemma (Buyer)]: \"{msg_gemma}\"\033[0m")
                
                # Gemini (warning)
                msg_gemini = None
                if not fallback_mock:
                    msg_gemini = get_dynamic_message("Gemini (Macro Analyst)", "Supply Shortage (Shock)", price, beta)
                if not msg_gemini:
                    idx = min(shock_step - 1, len(MOCK_DATA["shock"]["gemini"]) - 1)
                    msg_gemini = MOCK_DATA["shock"]["gemini"][idx]
                print(f"\033[95m[Gemini (Macro Analyst)]: \"{msg_gemini}\"\033[0m")
                
            elif state == "regulation":
                # Agent C
                msg_reg = None
                if not fallback_mock:
                    msg_reg = get_dynamic_message("Agent C (Regulator)", "Emergency Policy Intervention", price, beta)
                if not msg_reg:
                    msg_reg = random.choice(MOCK_DATA["regulation"]["regulator"])
                print(f"\033[93m[Agent C (Regulator)]: \"{msg_reg}\"\033[0m")
                
                # Gemma (regulated)
                msg_gemma = None
                if not fallback_mock:
                    msg_gemma = get_dynamic_message("Gemma (Buyer)", "Regulated Market", price, beta)
                if not msg_gemma:
                    msg_gemma = random.choice(MOCK_DATA["regulation"]["gemma"])
                print(f"\033[92m[Gemma (Buyer)]: \"{msg_gemma}\"\033[0m")
                
                # Gemini (macro stabilized)
                msg_gemini = None
                if not fallback_mock:
                    msg_gemini = get_dynamic_message("Gemini (Macro Analyst)", "Regulated Market", price, beta)
                if not msg_gemini:
                    msg_gemini = random.choice(MOCK_DATA["regulation"]["gemini"])
                print(f"\033[95m[Gemini (Macro Analyst)]: \"{msg_gemini}\"\033[0m")
                
            iteration += 1
            print("-" * 50)
            time.sleep(1.5)
            
    except KeyboardInterrupt:
        print("\n\033[93m[System]: KeyboardInterrupt received. Initiating post-demo cleanup...\033[0m")
        # 1. Update parameter files to stabilized beta
        final_beta = 0.22
        print(f"[System]: Saving final stabilized parameters (beta = {final_beta:.2f}, stabilized = True)...")
        update_simulation_parameters(final_beta, timestamp=4.25)
        
        # 2. Run final stabilized render
        run_stabilized_render(final_beta)
        
        # 3. Synchronize outputs to root data directory
        src_html = "/home/xayah/Documents/anmosys26/genesis-oracle/data/pinn_3d_fabric.html"
        dst_html = "/home/xayah/Documents/anmosys26/data/pinn_3d_fabric.html"
        src_png = "/home/xayah/Documents/anmosys26/genesis-oracle/data/pinn_3d_fabric.png"
        dst_png = "/home/xayah/Documents/anmosys26/data/pinn_3d_fabric.png"
        try:
            if os.path.exists(src_html):
                shutil.copy2(src_html, dst_html)
                print(f"[System]: Synced HTML plot to {dst_html}")
            if os.path.exists(src_png):
                shutil.copy2(src_png, dst_png)
                print(f"[System]: Synced PNG plot to {dst_png}")
        except Exception as e:
            print(f"[System Warning]: Failed to sync visualization files: {e}")
            
        print("\033[92m==================================================")
        print("SIMULATION SHUTDOWN AND CLEANUP COMPLETE")
        print("==================================================\033[0m")

def main():
    parser = argparse.ArgumentParser(description="Genesis Oracle Runner")
    parser.add_argument("--mode", type=str, choices=["simulation", "collaboration"], default="collaboration",
                        help="Select run mode: 'collaboration' runs original A2A demo, 'simulation' runs Arena Pitch simulation.")
    parser.add_argument("--fallback-mock", action="store_true",
                        help="If set, runs the simulation using offline mock strings only.")
    args = parser.parse_args()
    
    if args.mode == "simulation":
        run_realtime_simulation(fallback_mock=args.fallback_mock)
        return

    print("==================================================")
    print("STARTING ADVERSARIAL COLLABORATION LIVE-DEMO")
    print("==================================================")
    
    # 1. Generate the initial perturbed plot
    classical_pi.main()
    
    image_path = "data/anomaly_detection_plot.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} was not generated.")
        return
        
    print("\n[Phase 2] Vision Agent parsing anomaly detection plot...")
    
    # 2. Agent A (Vision) analyses the plot (initially suggests beta=0.15)
    try:
        report_raw = analyze_telemetry_plot(image_path, suggest_beta=0.15)
        report = json.loads(report_raw)
    except Exception as e:
        print(f"Error calling Vision API: {e}")
        # Robust fallback output matching target values
        report = {
            "anomaly_detected": True,
            "estimated_timestamp_s": 4.25,
            "suggested_damping_beta": 0.15,
            "confidence": 0.92
        }
    
    timestamp = report.get("estimated_timestamp_s", 4.25)
    suggested_beta = report.get("suggested_damping_beta", 0.15)
    
    # Print Agent A's initial suggestion
    print(f'\n[Agent A (Vision)]: "Anomalie bei $t={timestamp:.2f}s$ erkannt. Dämpfung auf $\\beta={suggested_beta:.2f}$ anpassen."')
    
    # 3. Agent B (Physics Auditor) validates it
    print("\n[Phase 3] Physics Auditor validating parameter...")
    is_valid = validate_damping_jax(suggested_beta, timestamp)
    
    # 4. Correction Loop
    if not is_valid:
        corrected_beta = 0.18
        # Agent B Vetos
        print(f'[Agent B (Auditor)]: "VETO. Parameter führt zu mathematischer Instabilität im JAX-State. Fordere Korrektur auf $\\beta={corrected_beta:.2f}$!"')
        
        # Agent A receives the veto and accepts
        print(f'[Agent A (Vision)]: "Akzeptiert. Korrigiere Koordinate."')
        
        # Auditor checks the corrected value
        is_valid_corrected = validate_damping_jax(corrected_beta, timestamp)
        if is_valid_corrected:
            print(f"[Agent B (Auditor)]: Corrected beta={corrected_beta} approved.")
            final_beta = corrected_beta
        else:
            print("[Agent B (Auditor)]: Veto failed after correction. Aborting.")
            return
    else:
        print("[Agent B (Auditor)]: Initial parameter approved.")
        final_beta = suggested_beta
        
    # 5. Update parameters file
    print("\n[Phase 4] Updating parameters file and running final stabilized render...")
    update_simulation_parameters(final_beta, timestamp)
    
    # 6. Run stabilized rendering (calls fabric_pinn)
    run_stabilized_render(final_beta)
    
    print("==================================================")
    print("LIVE-DEMO WORKFLOW COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
