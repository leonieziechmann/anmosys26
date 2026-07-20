import os
import sys
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.adk.agents.llm_agent import Agent

# Load environment variables manually
def load_env_manual():
    env_paths = [
        os.path.join(os.path.dirname(__file__), "..", ".env"),
        ".env",
        "genesis-oracle/.env"
    ]
    for path in env_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")

load_env_manual()
api_key = os.environ.get("GEMINI_API_KEY", "MOCK_API_KEY")

class AnomalyReport(BaseModel):
    anomaly_detected: bool = Field(description="Must be true if an anomaly/disturbance is detected in the plot")
    estimated_timestamp_s: float = Field(description="Estimated timestamp in seconds where the anomaly starts")
    suggested_damping_beta: float = Field(description="Suggested damping factor beta to apply to the oscillator")
    confidence: float = Field(description="Confidence score of the visual analysis")

def analyze_telemetry_plot(image_path: str, suggest_beta: float = 0.15) -> str:
    """
    Parses the anomaly detection plot using Gemini 3.5 Flash and returns the structured JSON report.
    """
    client = genai.Client(api_key=api_key)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    img = Image.open(image_path)
    
    prompt = (
        "Analyze this telemetry signal plot representing a damped harmonic oscillator. "
        "Locate any high-frequency stochastic noise, discontinuities, or abnormal amplitudes. "
        "Generate a report using the required JSON schema. "
        f"For the suggested damping factor, please recommend beta = {suggest_beta}."
    )
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=AnomalyReport,
        temperature=0.0
    )
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[img, prompt],
        config=config
    )
    return response.text.strip()

def adjust_reactor_temperature(delta_t: float) -> str:
    """
    Adjusts the core temperature of the reactor.

    Args:
        delta_t: The amount to increase or decrease the temperature in Kelvin.
    """
    new_temp = 300.0 + delta_t
    if new_temp > 350.0:
        return f"WARNING: Reactor overheated at {new_temp}K! Core breach imminent."
    return f"Success: Reactor stabilized at {new_temp}K."

root_agent = Agent(
    model='gemini-3.5-flash',
    name='observer_prime',
    description='A highly analytical agent specialized in managing physical reactor simulations.',
    instruction='You are Observer-Prime, a cold, highly logical AI overseeing a mathematical physics engine. Your primary goal is stabilization. You must always explain your reasoning clearly before taking action.',
    tools=[adjust_reactor_temperature]
)

