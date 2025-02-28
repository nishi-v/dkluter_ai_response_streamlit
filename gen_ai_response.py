from google import genai
from dotenv import load_dotenv
import os
from pathlib import Path
from PIL import Image
import re
import time
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, GoogleSearchRetrieval, DynamicRetrievalConfig

def gen_response(api: str, img: Image.Image, text: str) -> tuple[str, float]:
    # Generate AI Response
    client = genai.Client(api_key=api)
    model_id = "gemini-2.0-flash"

    google_search_tool = Tool(
        google_search = GoogleSearch(),
    )

    start = time.time()
    response = client.models.generate_content(
        model=model_id,
        contents=[img, text],
        config=GenerateContentConfig(
            temperature=0.0,
            seed=42,
            tools=[google_search_tool],
            response_modalities=["TEXT"],
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
    img_name :str = '61eoeu1UpRL._UF1000,1000_QL80_.jpg'
    img_file :str = os.path.join(dir, img_name)

    # Get API URL from environment variables
    GEMINI_API_KEY :str = os.environ["GEMINI_API_KEY"]

    # Image
    image :Image.Image = Image.open(img_file)

    # Prompt Text
    prompt_text :str = """
    Analyze the given image and generate a concise, accurate title. If there is a brand name or title, identify it.

    The description should focus only on the foreground object(s) and be no more than 80 words, providing relevant details about their characteristics, actions, or context while ignoring the background. If you are able to identify a known product in the image, please provide its details in the description rather than a simple captioning of what it visible. E.g., for a known book by an author, fill with when the book was released, the publisher, the genre, and a short summary of what the book is about rather than just captioning the text and image on the front cover of the book.    
    
    The tags should be structured hierarchically, first provide relevant tags starting from broad categories, moving on to more specific attributes. Avoid color-related visual tags.
    
    View this as an extension to OCR when text is present - extract all relevant details and supply them in the appropriate fields. Use the search tool to identify relevant attributes. For example, for an image of a book - fetch the title, author, publisher, ISBN number or other relevant details from the web that will be useful to know for the user.
    
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