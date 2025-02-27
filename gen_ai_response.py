from google import genai
from dotenv import load_dotenv
import os
from pathlib import Path
from PIL import Image
import json
import re

# Set current working directory
dir = Path(os.getcwd())

# Load environment variables from .env file
ENV_PATH = dir / '.env'
load_dotenv(ENV_PATH)

# Image file
img_dir = dir/'AssetImages'
img_name = 'image_DDBA3B38-37F3-4B0D-A359-4D3CECA1D42B.jpg'
img_file = img_dir/img_name

# Get API URL from environment variables
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

image = Image.open(img_file)

prompt_text = """
Analyze the given image and generate a concise, accurate title. If there is a brand name or title, identify it. The description should focus only on the foreground object(s) and be no more than 80 words, providing relevant details about their characteristics, actions, or context while ignoring the background. The tags should be structured hierarchically, first provide relevant tags starting from broad categories, moving on to more specific attributes. Avoid color-related visual tags. 

View this as an extension to OCR when text is present - extract all relevant details and supply them in the appropriate fields. For example, for an image of a book - identify the title, author, ISBN or other relevant details that are present and visible. 

Return the response strictly in the following JSON format, without any additional text, explanation, or preamble. Tags should be structured from most general to most specific, with abstract tags being placed at the end. 

One example format:
{
  "Message": "success message/Error message",
  "Data": {
    "title": "Modern Green Sofa with Elegant Vertical Stitching",
    "description": "Add a touch of sophistication and comfort to your living space with this stylish green sofa. Featuring a contemporary design, the sofa boasts soft, velvety upholstery with elegant vertical stitching on the backrest. The deep green tone complements various color schemes, making it perfect for modern or retro-inspired interiors. Supported by sturdy wooden legs, this sofa is as durable as it is visually appealing, offering both style and comfort for any room.",
    "tags": [
      "Furniture",
      "Livingroom",
      "Sofa",
      "Design"  
    ]
  },
  "Status": "ok/error"
}
"""

client = genai.Client(api_key=GEMINI_API_KEY)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image, prompt_text])
# print("API Response:", response.text)

if response.text:
    raw_response = response.text.strip()

# Remove markdown code block markers (```json ... ```)
cleaned_json = re.sub(r"^```json\n|\n```$", "", raw_response)

print(cleaned_json)
