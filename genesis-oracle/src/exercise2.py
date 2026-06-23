import os
import sys
import time

# Load environment variables
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

from google import genai
from google.genai import types
import jax
import jax.numpy as jnp

# JAX Mandelbrot Core Kernel - Fixed Loop Condition using loop counter i
@jax.jit
def mandelbrot_kernel(c, max_iters):
    def body_fn(val):
        z, count, active, i = val
        next_z = z**2 + c
        next_active = active & (jnp.abs(next_z) <= 2.0)
        next_count = jnp.where(next_active, count + 1, count)
        return next_z, next_count, next_active, i + 1

    def cond_fn(val):
        _, _, active, i = val
        return jnp.any(active) & (i < max_iters)

    z = jnp.zeros_like(c)
    count = jnp.zeros_like(c, dtype=jnp.int32)
    active = jnp.ones_like(c, dtype=jnp.bool_)
    
    _, final_counts, _, _ = jax.lax.while_loop(cond_fn, body_fn, (z, count, active, 0))
    return final_counts

def run_simulation(center_real, center_imag, zoom, resolution=400, max_iterations=500):
    # Map resolution coordinates to complex plane
    width, height = resolution, resolution
    r = jnp.linspace(center_real - 1.5 / zoom, center_real + 1.5 / zoom, width)
    i = jnp.linspace(center_imag - 1.5 / zoom, center_imag + 1.5 / zoom, height)
    R, I = jnp.meshgrid(r, i)
    C = R + 1j * I

    # Execute high-performance calculation on accelerator
    counts = mandelbrot_kernel(C.flatten(), max_iterations)
    counts = counts.reshape((height, width))

    # Calculate Shannon Entropy of escape times
    hist, _ = jnp.histogram(counts, bins=20)
    hist_prob = hist / jnp.sum(hist)
    hist_prob = jnp.where(hist_prob > 0, hist_prob, 1.0) # Avoid log(0)
    entropy = -jnp.sum(hist_prob * jnp.log(hist_prob))

    # Calculate Boundary Complexity (ratio of boundary pixels)
    boundary_pixels = jnp.sum((counts > 0) & (counts < max_iterations))
    boundary_ratio = boundary_pixels / (width * height)

    return counts, {
        "entropy": float(entropy),
        "boundary_complexity": float(boundary_ratio),
        "center_real": float(center_real),
        "center_imag": float(center_imag),
        "zoom": float(zoom),
        "max_iterations": int(max_iterations)
    }

# Define the tool wrapper that conforms to python typing
def simulate_mandelbrot(center_real: float, center_imag: float, zoom: float, max_iterations: int = 500) -> dict:
    """
    Runs a JAX-accelerated Mandelbrot simulation on the specified center coordinates and zoom factor.
    Returns visual complexity and Shannon entropy metrics.
    """
    _, metrics = run_simulation(center_real, center_imag, zoom, max_iterations=max_iterations)
    return metrics

def run_autonomous_agent(target_description):
    print("Initializing Gemini Client...")
    client = genai.Client(api_key=api_key)
    
    tools = [simulate_mandelbrot]
    
    prompt = (
        f"Your task is to navigate and find the target fractal detail: {target_description}.\n"
        f"You start at the global view (center_real=-0.5, center_imag=0.0, zoom=1.5).\n"
        f"Use the `simulate_mandelbrot` tool to explore and zoom in dynamically.\n"
        f"For each step, explain your reasoning (Thought), choose your coordinates and zoom level (Action),\n"
        f"and analyze the returned metrics (Observation).\n"
        f"Target Coordinates for Seahorse Valley: c ≈ -0.7436 + 0.1318i.\n"
        f"Your goal is to reach a zoom level of at least 15,000x (zoom >= 15000.0) at this valley.\n"
        f"Once you reach zoom >= 15000.0, stop calling tools and output your final summary."
    )
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    
    step = 0
    max_steps = 15
    model_name = "gemini-2.5-flash"
    
    print("Starting Autonomous Agent Loop...")
    while step < max_steps:
        step += 1
        print(f"\n=== Step {step} ===")
        
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=tools,
                system_instruction=(
                    "You are an autonomous fractal explorer agent. "
                    "You must zoom in step-by-step to locate Seahorse Valley (c ≈ -0.7436 + 0.1318i). "
                    "In each step, call `simulate_mandelbrot` to inspect the region. "
                    "Gradually increase the zoom factor (e.g. from 1.5, to 10, to 100, to 1000, and finally >= 15000). "
                    "Keep coordinate precision appropriate for your zoom level. "
                    "Stop and provide a summary once zoom >= 15000.0 is reached."
                )
            )
        )
        
        if response.text:
            print(f"Agent Thought/Response:\n{response.text}")
            
        function_calls = response.function_calls
        if not function_calls:
            print("No function calls requested by the model. Converged or stopped.")
            break
            
        contents.append(response.candidates[0].content)
        
        for call in function_calls:
            print(f"Tool Action requested: {call.name} with args: {call.args}")
            
            if call.name == "simulate_mandelbrot":
                args = dict(call.args)
                try:
                    center_real = float(args.get("center_real"))
                    center_imag = float(args.get("center_imag"))
                    zoom = float(args.get("zoom"))
                    max_iterations = int(args.get("max_iterations", 500))
                except Exception as e:
                    print(f"Error parsing tool arguments: {e}")
                    obs_content = types.Content(
                        role="tool",
                        parts=[types.Part.from_function_response(
                            name=call.name,
                            response={"error": str(e)}
                        )]
                    )
                    contents.append(obs_content)
                    continue
                
                metrics = simulate_mandelbrot(
                    center_real=center_real,
                    center_imag=center_imag,
                    zoom=zoom,
                    max_iterations=max_iterations
                )
                print(f"Tool Observation (Metrics): {metrics}")
                
                obs_content = types.Content(
                    role="tool",
                    parts=[types.Part.from_function_response(
                        name=call.name,
                        response=metrics
                    )]
                )
                contents.append(obs_content)
                
                if zoom >= 15000.0:
                    print("Target zoom level reached!")
            else:
                print(f"Unknown tool call: {call.name}")
                obs_content = types.Content(
                    role="tool",
                    parts=[types.Part.from_function_response(
                        name=call.name,
                        response={"error": "Unknown tool name"}
                    )]
                )
                contents.append(obs_content)
                
        # Brief sleep between steps (runs instantly locally)
        time.sleep(0.5)

if __name__ == "__main__":
    run_autonomous_agent("Seahorse Valley (c ≈ -0.7436 + 0.1318i) at zoom >= 15000x")
