import os
import sys
import json
import asyncio
import argparse

# -----------------------------------------------------------------------------
# Environment & Credentials Configuration
# -----------------------------------------------------------------------------
# Resolve .env path from genesis-oracle workspace if running in nix-shell
DOTENV_PATH = "/home/xayah/Documents/anmosys26/genesis-oracle/.env"
if os.path.exists(DOTENV_PATH):
    with open(DOTENV_PATH) as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key] = val.strip('"')

# Ensure we have our environment variables set
if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    print("Warning: GOOGLE_API_KEY/GEMINI_API_KEY not found in environment.")

from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.runners import InMemoryRunner
from google.genai import types

# -----------------------------------------------------------------------------
# Mock Tools for Local Fallback / Offline Mode
# -----------------------------------------------------------------------------
def check_bridge_status() -> str:
    """Check the real-time status of Spreeland bridges from the database.
    
    Returns:
        A JSON string containing the status of Burg and Cottbus bridges.
    """
    return json.dumps({
        "Burg_Bridge_01": {
            "status": "OPEN",
            "max_load_limit_tons": 40,
            "maintenance_completed_at": "2026-07-19T12:00:00Z"
        },
        "Cottbus_Main_Bridge": {
            "status": "CLOSED",
            "reason": "Automated Maintenance",
            "reopens_at": "2026-07-19T18:00:00Z"
        },
        "Spreewald_East_Bridge": {
            "status": "OPEN",
            "max_load_limit_tons": 30,
            "maintenance_completed_at": "2026-07-18T10:00:00Z"
        }
    })

# -----------------------------------------------------------------------------
# Sub-Agent Definitions (A2A Collaboration Pattern)
# -----------------------------------------------------------------------------
weather_predictor = Agent(
    model="gemini-2.5-flash",
    name="weather_predictor",
    description="A specialized weather predictor agent that forecasts Spreeland river levels and weather conditions.",
    instruction="""You are a Weather-Predictor Agent.
    Provide river level forecasts and weather trends.
    Current conditions: Burg river level is stable at 1.4m. Weather is clear. Cottbus expects light rain later today."""
)

supplier_agent = Agent(
    model="gemini-2.5-flash",
    name="supplier_agent",
    description="A local farmer cooperative agent for purchasing organic gherkins in bulk.",
    instruction="""You negotiate wholesale gherkin sales.
    You have a batch of 2 tons of organic gherkins available at Cottbus depot.
    Price is 1500 EUR per ton.
    If asked to authorize a purchase, confirm availability, agree on the total price (3000 EUR), and provide a cryptographic transaction authorization code (e.g. SHA-256 hash)."""
)

# -----------------------------------------------------------------------------
# Core Streaming Execution Loop (AG-UI/A2UI Pattern)
# -----------------------------------------------------------------------------
async def run_dispatch_flow(use_mock: bool):
    print("=" * 70)
    print("Spreeland Logistics Sync Coordinator Initializing...")
    print("=" * 70)
    
    # 1. Connect to Infrastructure Data via MCP or Local Fallback
    if use_mock:
        print("[System]: Operating in Mock Mode. Using offline tools.")
        tools = [check_bridge_status]
    else:
        print("[System]: Attempting connection to infrastructure data via MCP...")
        try:
            infra_tools = McpToolset(connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx", args=["-y", "@spreeland/bridge-mcp-server"],
                    env={"API_KEY": "SPREE_2026_SECRET"})
            ))
            tools = [infra_tools]
        except Exception as e:
            print(f"[System Warning]: MCP connection failed to initialize ({e}). Falling back to local tools.")
            tools = [check_bridge_status]

    # 2. Define the Dispatcher Agent
    dispatcher = Agent(
        model="gemini-2.5-flash",
        name="Spreeland_Dispatcher",
        description="Coordinates logistics, bridge checks, weather audits, and supplier purchases in Spreeland.",
        instruction="""You coordinate logistics in Spreeland. 
        1. Check bridge status via the bridge checking tool.
        2. Consult the weather_predictor agent for river conditions.
        3. Negotiate with the supplier_agent to purchase 2 tons of organic gherkins.
        4. Based on the bridge status and weather, determine if the route Burg -> Cottbus is safe and open.
        5. Output the final dispatch status in the A2UI JSON schema format representing a delivery card.
        Format your reasoning step-by-step and show the final A2UI schema output clearly.""",
        tools=tools,
        sub_agents=[weather_predictor, supplier_agent]
    )

    # 3. Handle Interactive Streaming (AG-UI pattern)
    runner = InMemoryRunner(agent=dispatcher)
    runner.auto_create_session = True
    
    prompt = (
        "Coordinate a dispatch of 2 tons of organic gherkins from Burg to Cottbus. "
        "Check bridge status, verify weather conditions with the weather predictor, "
        "and complete the purchase negotiation with the supplier agent. Finally, output the status card schema."
    )
    
    user_message = types.Content(role="user", parts=[types.Part(text=prompt)])
    
    print(f"\n🚀 Starting Live Dispatch Workflow.")
    print(f"Request: {prompt}\n")
    
    try:
        async with runner:
            async for event in runner.run_async(
                user_id="dispatch_operator_2026",
                session_id="session_spreeland_sync_01",
                new_message=user_message
            ):
                author = event.author or "System"
                
                # Check for streaming text content (thought process/reasoning)
                if event.content and event.content.parts:
                    text_parts = [p.text for p in event.content.parts if p.text]
                    if text_parts:
                        text = "".join(text_parts)
                        print(f"[{author}]: {text}", end="", flush=True)
                
                # Check for tool/sub-agent calls (A2A or MCP)
                func_calls = event.get_function_calls()
                if func_calls:
                    print()
                    for fc in func_calls:
                        print(f"\n⚙️  [{author}] calling tool/agent '{fc.name}' with args: {fc.args}")
                
                # Check for tool/sub-agent responses
                func_responses = event.get_function_responses()
                if func_responses:
                    print()
                    for fr in func_responses:
                        print(f"📥  [{author}] received response from '{fr.name}': {fr.response}")
                        
    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
        # If we failed running the real MCP because the server is not installed, retry in mock mode
        if not use_mock:
            print("\n[System]: Retrying execution in Mock Mode...")
            await run_dispatch_flow(use_mock=True)
            return

    print("\n" + "=" * 70)
    print("Spreeland Logistics Sync Finished.")
    print("=" * 70)

# -----------------------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------------------
def inject_shock_state(shock_name: str):
    if shock_name != "supply_shortage":
        print(f"[Dispatcher Error]: Unknown shock type '{shock_name}'. Only 'supply_shortage' is supported.")
        sys.exit(1)
        
    print("=" * 70)
    print(f"🚨  [Dispatcher]: INJECTING SHOCK '{shock_name}' into Simulation Parameters...")
    print("=" * 70)
    
    paths = [
        "/home/xayah/Documents/anmosys26/simulation_parameters.json",
        "/home/xayah/Documents/anmosys26/genesis-oracle/simulation_parameters.json"
    ]
    updated = False
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                data["supply_shortage"] = True
                with open(p, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"[Dispatcher]: Updated {os.path.basename(p)} -> supply_shortage: True")
                updated = True
            except Exception as e:
                print(f"[Dispatcher Error]: Failed to write to {p}: {e}")
                
    if updated:
        print("\n[Dispatcher Status]: Shock successfully injected. Game-loop watcher will trigger event horizon.")
    else:
        print("[Dispatcher Error]: No parameter files found to update.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spreeland Logistics Sync starter script.")
    parser.add_argument("--mock", action="store_true", help="Run with mock database tools (default when offline).")
    parser.add_argument("--inject-shock", type=str, help="Inject a disturbance/shock into the simulation.")
    args = parser.parse_args()
    
    if args.inject_shock:
        inject_shock_state(args.inject_shock)
    else:
        # Run the async loop
        asyncio.run(run_dispatch_flow(use_mock=args.mock))
