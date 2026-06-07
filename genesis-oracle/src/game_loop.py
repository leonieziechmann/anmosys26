import os
import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from sandbox_env import ThermalDampenerEnv

# 1. Define the regulatory Pydantic model schema
class ControlDecision(BaseModel):
    system_state: str = Field(description="Must be 'FREEZING', 'BOILING', or 'PERFECT'")
    adjustment_action: str = Field(description="Must be 'INCREASE', 'DECREASE', or 'HOLD'")
    delta_value: float = Field(description="The exact numerical change to apply to Kappa")
    confidence_score: float

def main():
    # Setup environment (highly volatile BOILING starting state)
    env = ThermalDampenerEnv(initial_kappa=12.0, initial_temp=120.0)
    
    # Initialize Gemini client
    api_key = os.environ.get("GEMINI_API_KEY", "MOCK_API_KEY")
    client = genai.Client(api_key=api_key)
    
    print("==================================================")
    print("STARTING THERMAL DAMPENER CLOSED-LOOP CONTROL GAME")
    print("==================================================")
    print(f"Initial State: {env.get_status_log()}\n")
    
    # Run loop for 5 consecutive turns
    for turn in range(1, 6):
        print(f"--- TURN {turn} ---")
        status_log = env.get_status_log()
        print(f"Env Output: {status_log}")
        
        # Prepare the prompt for Gemini
        prompt = (
            f"You are the Automated Thermal Regulator. The system telemetry is: {status_log}.\n"
            "Analyze the temperature log and make a control decision to guide the system "
            "to the PERFECT zone (between 20.0 and 30.0 K, optimal Kappa is 5.0).\n"
            "Respond using the required JSON schema."
        )
        
        # Call the API with structured output configuration
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ControlDecision,
            temperature=0.0
        )
        
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=config
        )
        
        # Parse the JSON response programmatically
        raw_json = response.text
        print(f"Raw API JSON Token: {raw_json.strip()}")
        
        try:
            # Validate through Pydantic to enforce the schema contract
            decision = ControlDecision.model_validate_json(raw_json)
            
            print(f"Validated Decision -> State: {decision.system_state} | "
                  f"Action: {decision.adjustment_action} | "
                  f"Delta Kappa: {decision.delta_value:+.2f} | "
                  f"Confidence: {decision.confidence_score:.2%}")
            
            # Apply the delta to Kappa in the sandbox environment
            new_temp, new_kappa = env.step(decision.delta_value)
            print(f"New State: {env.get_status_log()}\n")
            
        except Exception as e:
            print(f"Error validating or executing decision: {e}")
            break
            
    print("==================================================")
    print("GAME LOOP COMPLETED")
    print(f"Final State: {env.get_status_log()}")
    print("==================================================")

if __name__ == "__main__":
    main()
