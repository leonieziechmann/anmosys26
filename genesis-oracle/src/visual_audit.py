import os
from google import genai
from PIL import Image

def main():
    api_key = os.environ.get("GEMINI_API_KEY", "MOCK_API_KEY")
    client = genai.Client(api_key=api_key)
    
    image_path = os.path.join("data", "audit_target.png")
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist. Run generate_signals.py first.")
        return
        
    print(f"Loading image {image_path}...")
    image = Image.open(image_path)
    
    prompt = (
        "Act as a Visual Detective. Analyze this waveform signal. "
        "Find the visual anomaly, guess the exact pixel/X-axis region where the malfunction happened, "
        "and write a short, funny poem mocking the engineering team that allowed this bug to pass."
    )
    
    print("Sending multimodal request to Gemini...")
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[image, prompt]
    )
    
    print("\n--- Visual Detective Report ---")
    print(response.text)
    print("-------------------------------")

if __name__ == "__main__":
    main()
