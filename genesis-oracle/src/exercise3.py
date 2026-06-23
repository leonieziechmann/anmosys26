import os
import sys
import yaml
import json
import time

# Manual .env loader
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

class GemmaSkillLoader:
    def __init__(self, skill_dir):
        self.skill_dir = skill_dir
        self.instructions = ""
        self.metadata = {}
        self.schemas = []
        self.load_skill()
        
    def load_skill(self):
        # 1. Parse YAML frontmatter and Markdown body from SKILL.md
        skill_md_path = os.path.join(self.skill_dir, "SKILL.md")
        if not os.path.exists(skill_md_path):
            raise FileNotFoundError(f"SKILL.md not found at {skill_md_path}")
            
        with open(skill_md_path, "r") as f:
            content = f.read()
            
        if content.strip().startswith("---"):
            parts = content.strip().split("---", 2)
            if len(parts) >= 3:
                self.metadata = yaml.safe_load(parts[1])
                self.instructions = parts[2].strip()
            else:
                self.instructions = content.strip()
        else:
            self.instructions = content.strip()
            
        # 2. Load JSON schemas from the tools/ directory
        tools_dir = os.path.join(self.skill_dir, "tools")
        if os.path.exists(tools_dir):
            for file in os.listdir(tools_dir):
                if file.endswith(".json"):
                    schema_path = os.path.join(tools_dir, file)
                    with open(schema_path, "r") as sf:
                        self.schemas.append(json.load(sf))
        
        # 3. Dynamic import of the solver scripts
        scripts_dir = os.path.join(self.skill_dir, "scripts")
        if os.path.exists(scripts_dir):
            if scripts_dir not in sys.path:
                sys.path.append(scripts_dir)

def boot_agent_from_skill(skill_dir):
    print("Loading Gemma-Skill...")
    loader = GemmaSkillLoader(skill_dir)
    print(f"Skill Loaded: {loader.metadata.get('name')}")
    print(f"Description: {loader.metadata.get('description')}")
    print("Loaded Tool Schemas:")
    for s in loader.schemas:
        print(f" - {s.get('name')}: {s.get('description')}")
        
    # Dynamically import the simulate_mandelbrot function from the skill solver script
    from mandelbrot_solver import simulate_mandelbrot
    
    # Initialize the client
    client = genai.Client(api_key=api_key)
    
    # Setup tools for modern google-genai client
    tools = [simulate_mandelbrot]
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="Please start exploration and find Seahorse Valley at zoom >= 15000.")]
        )
    ]
    
    step = 0
    max_steps = 15
    model_name = "gemini-2.5-flash"
    
    print("\nStarting Autonomous Optimization Loop using loaded instructions...")
    while step < max_steps:
        step += 1
        print(f"\n=== Skill Step {step} ===")
        
        # Call generate content with skill instructions as the system instruction
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=tools,
                system_instruction=loader.instructions
            )
        )
        
        if response.text:
            print(f"Agent Response:\n{response.text}")
            
        function_calls = response.function_calls
        if not function_calls:
            print("Finished exploration.")
            break
            
        contents.append(response.candidates[0].content)
        
        for call in function_calls:
            print(f"Skill Action: {call.name} with args: {call.args}")
            if call.name == "simulate_mandelbrot":
                args = dict(call.args)
                try:
                    center_real = float(args.get("center_real"))
                    center_imag = float(args.get("center_imag"))
                    zoom = float(args.get("zoom"))
                    max_iterations = int(args.get("max_iterations", 500))
                except Exception as e:
                    print(f"Error parsing args: {e}")
                    contents.append(types.Content(
                        role="tool",
                        parts=[types.Part.from_function_response(name=call.name, response={"error": str(e)})]
                    ))
                    continue
                    
                metrics = simulate_mandelbrot(
                    center_real=center_real,
                    center_imag=center_imag,
                    zoom=zoom,
                    max_iterations=max_iterations
                )
                print(f"Skill Observation: {metrics}")
                
                contents.append(types.Content(
                    role="tool",
                    parts=[types.Part.from_function_response(name=call.name, response=metrics)]
                ))
            else:
                contents.append(types.Content(
                    role="tool",
                    parts=[types.Part.from_function_response(name=call.name, response={"error": "unknown tool"})]
                ))
        
        time.sleep(0.5)

if __name__ == "__main__":
    skill_path = "/home/xayah/Documents/anmosys26/skills/mandelbrot_explorer"
    boot_agent_from_skill(skill_path)
