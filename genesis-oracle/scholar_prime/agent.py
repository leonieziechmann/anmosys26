import os
import subprocess
import certifi
from google.adk.agents.llm_agent import Agent

def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Searches the arXiv scientific literature database for relevant publications.

    Args:
        query: The search query string (e.g., 'thermodynamic simulation parameters for advanced fission reactors').
        max_results: The maximum number of results to return (defaults to 5).

    Returns:
        A JSON string containing the search results (list of papers with titles, abstracts/summaries, and DOIs/URLs).
    """
    # Path to the arXiv search script in the cloned science-skills repo
    script_path = "/home/xayah/Documents/anmosys26/science-skills/skills/literature_search_arxiv/scripts/search_arxiv.py"
    
    # Configure the environment to use certifi's SSL bundle
    env = os.environ.copy()
    env["SSL_CERT_FILE"] = certifi.where()
    
    cmd = [
        "uv", "run", script_path,
        "--query", query,
        "--max_results", str(max_results)
    ]
    
    try:
        # Run command and capture output
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        return res.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing search_arxiv CLI: {e.stderr}\nOutput: {e.output}"

root_agent = Agent(
    model='gemini-2.5-flash',
    name='scholar_prime',
    description='An academic research agent specialized in querying scientific databases and extracting material parameters.',
    instruction=(
        'You are Scholar-Prime, a professional academic research agent. Your core expertise is '
        'searching scientific literature databases, evaluating abstract relevance for physical '
        'and thermodynamic parameters, extracting relevant simulation formulas or coefficients, '
        'and presenting them systematically. Whenever you present findings from a paper, you '
        'MUST always state the DOI or direct URL reference. Maintain high academic rigor and '
        'objective reporting in all responses.'
    ),
    tools=[search_arxiv]
)
