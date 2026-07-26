import os
import sys

def main():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    print("Initializing Gemini Client...")
    
    prompt = (
        "Explain the difference between a stateful NumPy random generation process "
        "and a stateless JAX PRNG split operation in exactly one highly sarcastic sentence."
    )
    print(f"Sending prompt: {prompt}\n")
    
    if api_key and api_key != "IHRE_API_KEY_VARIABLE":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt
            )
            print("Response from Oracle:")
            print(response.text)
            return
        except Exception as e:
            print(f"[API Warning] Could not connect to Gemini API ({e}). Falling back to mock oracle response:")
    else:
        print("[Info] No active GOOGLE_API_KEY set. Using local Oracle response:")

    print("Response from Oracle:")
    print('"NumPy keeps a global mutable state like a sloppy notebook, while JAX explicitly splits PRNG keys like a paranoid mathematician demanding proof of purity at every step."')

if __name__ == "__main__":
    main()

