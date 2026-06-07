import os
import sys
from google import genai

def main():
    # Retrieve the API key from environment
    api_key = os.environ.get("GEMINI_API_KEY", "MOCK_API_KEY")
    
    print("Initializing Gemini Client...")
    client = genai.Client(api_key=api_key)
    
    prompt = (
        "Explain the difference between a stateful NumPy random generation process "
        "and a stateless JAX PRNG split operation in exactly one highly sarcastic sentence."
    )
    
    print(f"Sending prompt: {prompt}\n")
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt
    )
    
    print("Response from Oracle:")
    print(response.text)

if __name__ == "__main__":
    main()
