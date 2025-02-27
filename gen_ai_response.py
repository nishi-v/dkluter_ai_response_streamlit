from google import genai
from dotenv import load_dotenv
import os
from pathlib import Path
from PIL import Image
import re
from google.genai import types
import time

def gen_response(api: str, img: Image.Image, text: str) -> tuple[str, float]:
    # Generate AI Response
    client = genai.Client(api_key=api)
    start = time.time()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[img, text],
        config=types.GenerateContentConfig(
            temperature=0.000001
        ))
    end = time.time() - start

    return response.text, end #type: ignore

def main():
    # Set current working directory
    dir = Path(os.getcwd())

    # Load environment variables from .env file
    ENV_PATH :Path= dir / '.env'
    load_dotenv(ENV_PATH)

    # Image file
    img_dir :str = os.path.join(dir, 'AssetImages')
    img_name :str = 'image_DDBA3B38-37F3-4B0D-A359-4D3CECA1D42B.jpg'
    img_file :str = os.path.join(img_dir, img_name)

    # Get API URL from environment variables
    GEMINI_API_KEY :str = os.environ["GEMINI_API_KEY"]

    # Image
    image :Image.Image = Image.open(img_file)

    # Prompt Text
    prompt_text :str = """
    Analyze the given image and generate a concise, accurate title. If there is a brand name or title, identify it. 

    The description should focus only on the foreground object(s) and be no more than 80 words, providing relevant details about their characteristics, actions, or context while ignoring the background. 

    The tags should be structured hierarchically, first provide relevant tags starting from broad categories, moving on to more specific attributes. Avoid color-related visual tags. 

    View this as an extension to OCR when text is present - extract all relevant details and supply them in the appropriate fields. 

    Return the response strictly in the following JSON format, without any additional text, explanation, or preamble.

    One example format:
    {
      "Data": {
        "title": "Modern Green Sofa with Elegant Vertical Stitching",
        "description": "Add a touch of sophistication and comfort to your living space with this stylish green sofa. Featuring a contemporary design, the sofa boasts soft, velvety upholstery with elegant vertical stitching on the backrest. The deep green tone complements various color schemes, making it perfect for modern or retro-inspired interiors. Supported by sturdy wooden legs, this sofa is as durable as it is visually appealing, offering both style and comfort for any room.",
        "tags": [
          "Furniture",
          "Livingroom",
          "Sofa",
          "Design"  
        ]
      }
    }
    """

    # Get generated response and time taken
    response_data, time_taken = gen_response(GEMINI_API_KEY, image, prompt_text)
    if response_data:
        raw_response = response_data.strip()

    # Remove markdown code block markers (```json ... ```)
    cleaned_json :str = re.sub(r"^```json\n|\n```$", "", raw_response)

    print(f"Response: \n{cleaned_json}")

    print(f"Time Taken to generate response: {time_taken}")

if __name__ == '__main__':
    main()