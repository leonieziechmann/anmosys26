import os
import sys
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Ensure genesis-oracle is in the python path
sys.path.append("/home/xayah/Documents/anmosys26/genesis-oracle")

# Load environment variables from .env
dotenv_path = "/home/xayah/Documents/anmosys26/genesis-oracle/.env"
if os.path.exists(dotenv_path):
    with open(dotenv_path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k] = v.strip('"')

from google import genai
from google.genai import types
from scholar_prime.agent import search_arxiv

# Define Pydantic models for structured output
class SimulationParameter(BaseModel):
    name: str = Field(description="The name of the physical/thermodynamic parameter.")
    value: str = Field(description="The value, range, or mathematical order of the parameter.")
    unit: str = Field(description="The unit of measurement (e.g., W/mK, m^2K/W).")
    context: str = Field(description="The physical context in which this parameter applies.")

class ExtractedData(BaseModel):
    paper_title: str = Field(description="Title of the research paper.")
    doi: str = Field(description="DOI of the paper.")
    url: str = Field(description="Direct URL or PDF link to the paper.")
    parameters: List[SimulationParameter] = Field(description="List of extracted simulation parameters.")

def extract_parameters_from_text(text: str) -> dict:
    """
    Extracts thermodynamic and physical simulation parameters from the given text
    using the Gemini API and returns them as a structured dictionary.

    Args:
        text: The source text (e.g., paper abstract) from which to extract parameters.

    Returns:
        A dictionary containing the extracted parameters, including their names, values,
        units, contexts, and the source paper reference.
    """
    client = genai.Client()
    
    prompt = (
        f"Extract all thermodynamic/physical parameters, constants, and simulation coefficients "
        f"from the following abstract. Return them in the requested JSON structure:\n\n{text}"
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractedData,
            system_instruction=(
                "You are an academic parameter extraction assistant. Carefully identify physical constants, "
                "interfacial resistance values, thermal conductivity bounds, and material specifications. "
                "Format them exactly in the JSON schema."
            )
        )
    )
    return json.loads(response.text)

def main():
    query = "UO2/BeO interfacial thermal resistance and its effect on fuel thermal conductivity"
    print(f"Running literature search on arXiv for query: '{query}'...")
    
    # Run the arXiv search tool
    search_results_str = search_arxiv(query=query, max_results=1)
    
    try:
        search_results = json.loads(search_results_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse search results as JSON: {e}")
        print(f"Search results content: {search_results_str}")
        sys.exit(1)
        
    if not search_results.get("papers"):
        print("No papers found matching the query.")
        sys.exit(1)
        
    # Get the most relevant paper (first one in results)
    paper = search_results["papers"][0]
    title = paper.get("title")
    abstract = paper.get("summary")
    doi = paper.get("doi", "N/A")
    pdf_url = paper.get("pdf_url", "N/A")
    
    print(f"\nMost relevant paper identified:")
    print(f"Title: {title}")
    print(f"DOI: {doi}")
    
    print("\nExtracting thermodynamic parameters from abstract...")
    extracted_dict = extract_parameters_from_text(abstract)
    
    # Ensure references are filled if the model missed them
    if not extracted_dict.get("paper_title") or extracted_dict["paper_title"] == "N/A":
        extracted_dict["paper_title"] = title
    if not extracted_dict.get("doi") or extracted_dict["doi"] == "N/A":
        extracted_dict["doi"] = doi
    if not extracted_dict.get("url") or extracted_dict["url"] == "N/A":
        extracted_dict["url"] = pdf_url
        
    output_file = "/home/xayah/Documents/anmosys26/genesis-oracle/simulation_parameters.json"
    print(f"\nWriting extracted parameters to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(extracted_dict, f, indent=2)
        
    print("Done! Here is the extracted JSON:")
    print(json.dumps(extracted_dict, indent=2))

if __name__ == "__main__":
    main()
