from google import genai
from dotenv import load_dotenv
import os
from pathlib import Path
from PIL import Image
import re
import time
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import pandas as pd
import json
import sys

def get_image(img_file: str) -> Image.Image:
    try:
        img = Image.open(img_file)
        return img
    except Exception as e:
        print(f"Error fetching {img_file}: {e}")
        return None # type: ignore

def gen_batch_response(api: str, img: Image.Image, types: list) -> tuple[dict, float]:
    # Generate AI Response
    client = genai.Client(api_key = api)
    model_id = "gemini-2.0-flash"

    google_search_tool = Tool(
        google_search = GoogleSearch(),
    )

    prompt_text = """
        Analyze the given image and the types given with it, and generate a concise, accurate title. If there is a brand name or title, identify it with a max length of 100 characters.
    
        The description should focus only on the foreground object(s) and be no more than 80 words, providing relevant details about their characteristics, actions, or context while ignoring the background. If you are able to identify a known product in the image, please provide its details in the description rather than a simple captioning of what it visible. E.g., for a known book by an author, mention when the book was released, the publisher, the genre, and a short summary of what the book is about rather than just captioning the text and image on the front cover of the book.    
        
        The tags should be structured hierarchically, first provide relevant tags starting from broad categories, moving on to more specific attributes. Avoid color-related visual tags. Max limit to tags is 50 characters.
        
        ALWAYS generate relevant field name, type and value based on the objects and types provided. Field names should ALWAYS be in list format.
    
        Field should ALWAYS be present and NON EMPTY. If a field value cannot be determined, provide the best possible field. If possible add LOCATION type Field if found.

        Each field **must** have:
        - A meaningful 'field_name' (max 50 characters).
        - A valid 'field_type' (TEXT, NUMBER, DATE, LOCATION).
        - A **non-empty 'field_value'** (max 500 characters).

        Ensure that **no field_name, field_type, or field_value is missing**.
            
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
    prompt_text += f'\nThese are the tag categories to which the object belongs, which you can use as an additional reference to narrow down your search domain: "types": {types}'
    # print(prompt_text)

    start_time = time.time()
    response = client.models.generate_content(
        model = model_id,
        contents = [img, prompt_text],
        config=GenerateContentConfig(
            temperature = 0.0,
            seed = 42,
            tools = [google_search_tool],
            response_modalities = ["TEXT"],
        ))  
    end = time.time() - start_time

    if response.text:
        raw_response = response.text.strip()
        # Remove markdown code block markers (```json ... ```)
        cleaned_json = re.sub(r"^```json\n|\n```$", "", raw_response)
        try:
            return json.loads(cleaned_json), end
        except json.JSONDecodeError:
            print(f"Error parsing JSON for image:")
            return {"Data": {"title": "", "description": "", "tags": [], "fields": []}}, round(end, 2)
    else:
        return {"Data": {"title": "", "description": "", "tags": [], "fields": []}}, round(end, 2)

def main():
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"Received csv_file: {csv_file}")
    else:
        print("CSV file not provided. Exiting...")
        sys.exit(1)  # Exit the script with an error code

    print(f"Generating Responses...")
    # Set current working directory
    dir = Path(os.getcwd())

    # Load environment variables from .env file
    ENV_PATH :Path= dir / '.env'
    load_dotenv(ENV_PATH)

    # Image file
    img_dir :str = os.path.join(dir, 'asset_images')

    # Get API URL from environment variables
    GEMINI_API_KEY :str = os.environ["GEMINI_API_KEY"]

    # print(csv_file)
    df = pd.read_csv(csv_file)  
    # print(df)

    results = []

    for index, row in df.iterrows():
        img_name = row['Image']

        img_file = os.path.join(img_dir, img_name)

        types = row['Types'].split(', ') if isinstance(row['Types'], str) else []

        image = get_image(img_file)
        print(f"Processing {index+1}: {img_name}") #type:ignore
        try:
            # Generate Response
            response_data, time_taken = gen_batch_response(GEMINI_API_KEY, image, types)

            print(f"Response: \n{response_data}")
            print(f"Time taken: {time_taken:.2f} seconds")

            # Extract Data
            data = response_data.get("Data", {})
            title = data.get("title", "")
            description = data.get("description", "")
            tags = ', '.join(data.get("tags", []))

            # Process fields into a single field string
            fields_list = data.get("fields", [])
            fields_data = []
            for field in fields_list:
                field_name = field.get("field_name", "")
                field_type = field.get("field_type", "")
                field_values = field.get("field_value", [])
                # Ensure field_value is treated correctly
                if isinstance(field_values, list):
                    # Convert list of single characters into a proper string if needed
                    field_value_str = ', '.join([''.join(v) if isinstance(v, list) else str(v) for v in field_values])
                else:
                    field_value_str = str(field_values)  # Handle non-list cases safely

                fields_data.append(f"{field_name} ({field_type}): {field_value_str}")

            fields = ', '.join(fields_data)

            # Store the complete JSON response
            json_response = json.dumps(response_data)

            # Storing all data in a single result
            results.append({
                "Image": img_name,
                "Types": ", ".join(types),
                "Title": title,
                "Description": description,
                "Tags": tags,
                "Fields": fields,
                "Time": time_taken,
                "Json Response": json_response
            })

        except Exception as e:
            print(f"Error in getting response for {img_name}: {e}")
            # Adding an empty result
            results.append({
                "Image": img_name,
                "Types": ", ".join(types),
                "Title": "",
                "Description": "",
                "Tags": "",
                "Fields": f"Error: {str(e)}",
                "Time": 0.0,
                "Json Response": ""
            })
            continue

    # Create a new DataFrame with results to ensure proper types
    results_df = pd.DataFrame(results)

    # Format the Time column to have 2 decimal places
    results_df['Time'] = results_df['Time'].apply(lambda x: f"{x:.2f}")
    
    # Save the results to CSV, replacing the original file
    results_df.to_csv(csv_file, index=False)

    print(f"Processing complete! Results saved in {csv_file}.")

if __name__ == '__main__':
    main()