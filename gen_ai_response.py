from google import genai
from dotenv import load_dotenv
import os
from pathlib import Path
from PIL import Image
import re
import time
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

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

def count_token(api: str, text: str):
    client = genai.Client(api_key=api)
    response = client.models.count_tokens(
        model='gemini-2.0-flash',
        contents=text,
    )
    return response

def main():
    # Set current working directory
    dir = Path(os.getcwd())

    # Load environment variables from .env file
    ENV_PATH :Path= dir / '.env'
    load_dotenv(ENV_PATH)

    # Image file
    img_dir :str = os.path.join(dir, 'AssetImages')
    img_name :str = 'image_2BEBA16C-F7A1-4A7A-8EE7-430D8B554CA1.jpg'
    img_file :str = os.path.join(img_dir, img_name)

    # Get API URL from environment variables
    GEMINI_API_KEY :str = os.environ["GEMINI_API_KEY"]

    # Image
    image :Image.Image = Image.open(img_file)

    # Type List
    type_list = ["Vehicle", "Tyre"]

    # Prompt Text
    prompt_text : str= f'"types": {type_list}' + """
    Analyze the given image and the types given with it, and generate a concise, accurate title. If there is a brand name or title, identify it with a max length of 100 characters.

    The description should focus only on the foreground object(s) and be no more than 80 words, providing relevant details about their characteristics, actions, or context while ignoring the background. If you are able to identify a known product in the image, please provide its details in the description rather than a simple captioning of what it visible. E.g., for a known book by an author, mention when the book was released, the publisher, the genre, and a short summary of what the book is about rather than just captioning the text and image on the front cover of the book.    
    
    The tags should be structured hierarchically, first provide relevant tags starting from broad categories, moving on to more specific attributes. Avoid color-related visual tags. Max limit to tags is 50 characters.
    
    Always generate relevant field name, type and value based on the objects and types provided. Field names should ALWAYS be in list format.

    Field name with antotal limit of 50 characters. Field Value with max length of 500 characters. Field types can only be of TEXT, NUMBER, DATE, LOCATION.

    View this as a smarter OCR when text is present - extract all relevant details and supply them in the appropriate fields. Use the search tool to identify relevant attributes. In case of a known product, identify model number, SKU, etc. E.g., for an image of a book - make sure to always search and fetch the title, author, publisher, ISBN number. Then, provide other relevant details from the web that will be useful to know for the user. Provide fields with values as objects within "tags" as a key-value pair.
    
    Return the response strictly in the following JSON format, without any additional text, explanation, or preamble.
    
    Example format:
    {
    "Data": {
        "title": "To Kill a Mockingbird by Harper Lee",
        "description": "A gripping, heart-wrenching, and wholly remarkable tale of coming-of-age in a South poisoned by virulent prejudice. It views a world of great beauty and savage inequities through the eyes of a young girl, as her father—a crusading local lawyer—risks everything to defend a black man unjustly accused of a terrible crime.",
        "tags": [
        "Book",
        "Classic",
        "Literature",
        "Novel",
        "Justice"
        ],
        "fields": [
            {
                "field_name": "ISBN",
                "field_type": "number",
                "field_value": ["9780061120084"]
            },
            {
                "field_name": "Genre",
                "field_type": "text",
                "field_value": ["Fiction"]
            }
            ]}
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

    # # Get total no. of tokens
    # response_tokens = count_token(GEMINI_API_KEY, prompt_text)
    # print(f"Total no. of tokens: {response_tokens}")

if __name__ == '__main__':
    main()