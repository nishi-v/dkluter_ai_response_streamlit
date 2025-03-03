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
import argparse
import glob

def get_image(img_file: str) -> Image.Image:
    try:
        img = Image.open(img_file)
        return img
    except Exception as e:
        print(f"Error fetching {img_file}: {e}")
        return None # type: ignore

def count_tokens(api_key:str, prompt_text:str) -> None:
    client = genai.Client(api_key = api_key)

    token_count = client.models.count_tokens(
        model='gemini-2.0-flash',
        contents=prompt_text,
    )
    print(f"Prompt tokens: {token_count}")

def gen_batch_response(api: str, img: Image.Image, types: list, prompt_text: str) -> tuple[dict, float]:
    # Generate AI Response
    client = genai.Client(api_key = api)
    model_id = "gemini-2.0-flash"

    google_search_tool = Tool(
        google_search = GoogleSearch(),
    )

    prompt_text = prompt_text
    
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

def get_next_filename(dir: str) -> str:
    existing_files = glob.glob(os.path.join(dir, "output_*.csv"))  
    next_num = len(existing_files) + 1  
    return os.path.join(dir, f"output_{next_num:05d}.csv")

def main():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-f", "--file", required=False, help="Name of the CSV file")
    parser.add_argument("-t", "--token", required=False, action="store_true", help="Include it to count tokens")
    parser.add_argument("-T", "--type", required=False, help="Define types if only count token is required")
    parser.add_argument("-i", "--img", required=False, help="Name of single image for response generation")

    args = parser.parse_args()

    csv_file = args.file
    should_count_tokens = args.token # Will be true if -t is passed, otherwise false.
    object_type = args.type # Give type of object, if only you want token count to be generated. No response will be generated.
    single_img_file = args.img # Name of single imagey ou want to generate response for.

    if csv_file is not None:
        print(f"Received csv file: {csv_file}")
        if should_count_tokens:
            print("Token counts will be generated.")
        else:
            print("Token count will not be generated.")
    elif csv_file is None and object_type is not None:
        if single_img_file is None:
            print(f"Only token count will be generated.")
        elif single_img_file is not None:
            print(f"Received image: {single_img_file}")
            if should_count_tokens:
                print("Token count will be generated.")
            else:
                print("Token count will not be generated.")
    elif csv_file is None and object_type is None:
        print("CSV file or Types is not provided. Exiting...")
        sys.exit(1)
    
    prompt_text = """
        Analyze the given image and types to generate: title, description, tags, fields. Title should be concise, accurate. If there is a brand name or title, identify it. Max length 100 characters.
        
        Description should focus only on the foreground object(s) and be no more than 80 words, providing relevant details about their characteristics, actions, or context while ignoring the background. If you are able to identify a known product in the image, provide its details rather than a simple captioning of what is visible. E.g., for a known book by an author, mention when the book was released, publisher, genre, and short summary of what the book is about rather than just captioning the text and image on the front cover of the book. Avoid using demonstrative pronouns ('this is', 'these are').  
            
        Tags should be structured hierarchically. First provide relevant tags starting from broad categories, moving on to  specific attributes. Tag max length is 50 characters. ALWAYS generate at least 3 tags.
            
        Avoid color-related visual tags and fields. 
            
        ALWAYS generate relevant field name, type, value based on the objects and types provided. Field names should ALWAYS be in list format.
        
        Field should ALWAYS be present and NON EMPTY. If field value can't be determined, provide best possible field. Field type can be: TEXT, NUMBER, DATE, LOCATION. Categorize alphanumeric values ('12 AB') and numbers with units ('15 mm') as TEXT. Numeric values ('123 45') should have whitespace removed and be categorized as NUMBER. If possible, add LOCATION type Field.

        Each field must have: Meaningful 'field_name' (max 50 characters), Valid 'field_type', Non-empty 'field_value' (max 500 characters)
                
        Use search tool to identify relevant attributes. In case of a known product, identify model number, SKU, etc. E.g., for image of a book - make sure to always search and fetch the title, author, publisher, ISBN number. Then, provide other relevant details from the web that will be useful to know for the user. 
            
        Return the response strictly in following JSON format, without additional text, explanation, or preamble:
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
                    "field_type": "NUMBER",
                    "field_value": ["9780061120084"]
                },
                {
                    "field_name": "Genre",
                    "field_type": "TEXT",
                    "field_value": ["Fiction"]
                }
                ]}
        }
        """ 
    
    # Set current working directory
    dir = Path(os.getcwd())

    # Load environment variables from .env file
    ENV_PATH :Path= dir / '.env'
    load_dotenv(ENV_PATH)

    # Image file
    img_dir :str = os.path.join(dir, 'asset_images')

    csv_file_dir = os.path.join(dir, "csv_files")

    # Get API URL from environment variables
    GEMINI_API_KEY :str = os.environ["GEMINI_API_KEY"]

    if csv_file is not None:
        print(f"Generating Responses...")

        csv_file_path = os.path.join(csv_file_dir, csv_file)

        # print(csv_file)
        df = pd.read_csv(csv_file_path)  
        df = df[df['Image'] != 'SUMMARY']
        # print(df)

        results = []
        total_time = 0
        successful_images = 0
        total_images = len(df)

        for index, row in df.iterrows():
            img_name = row['Image']
            if img_name == 'SUMMARY':
                continue

            img_file = os.path.join(img_dir, img_name)

            types = row['Types'].split(', ') if isinstance(row['Types'], str) else []

            image = get_image(img_file)
            print(f"Processing {index+1}: {img_name}")#type:ignore 
            try:
                prompt_text += f'\nThese are the tag categories to which the object belongs, which you can use as an additional reference to narrow down your search domain: "types": {types}'
                # Generate Response
                response_data, time_taken = gen_batch_response(GEMINI_API_KEY, image, types, prompt_text)

                # Add to total time
                total_time += time_taken
                successful_images += 1

                print(f"Response: \n{response_data}")
                print(f"Time taken: {time_taken:.2f} seconds")
                if should_count_tokens:
                    count_tokens(GEMINI_API_KEY, prompt_text)

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

        # Calculating average time
        avg_time = total_time/ successful_images if successful_images > 0 else 0

        # Create a new DataFrame with results to ensure proper types
        results_df = pd.DataFrame(results)

        # Format the Time column to have 2 decimal places
        results_df['Time'] = results_df['Time'].apply(lambda x: f"{x:.2f}")

        results_df = results_df[results_df['Image'] != 'SUMMARY']

        # Add summary statistics to the end of the DataFrame
        summary_df = pd.DataFrame([{
            "Image": "SUMMARY",
            "Types": f"Total Images: {total_images}",
            "Title": f"Successfully Processed: {successful_images}",
            "Description": f"Failed: {total_images - successful_images}",
            "Tags": f"Total: {total_time:.2f}s",
            "Fields": f"Average: {avg_time:.2f}s",
            "Time": "",
            "Json Response": ""
        }])

        # Combine the results with the summary
        results_df = pd.concat([results_df, summary_df], ignore_index=True)

        # Save the results to CSV, replacing the original file
        results_df.to_csv(csv_file_path, index=False)

        # Print summary statistics
        print(f"SUMMARY:")
        print(f"Total Images: {total_images}")
        print(f"Successfully Processed: {successful_images}")
        print(f"Failed: {total_images - successful_images}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Average Time per Image: {avg_time:.2f} seconds")
        
        print(f"Processing complete! Results saved in {csv_file_path}.")

    elif csv_file is None and object_type is not None:
        if single_img_file is None:
            prompt_text += f'\nThese are the tag categories to which the object belongs, which you can use as an additional reference to narrow down your search domain: "types": {object_type}' 
            print(prompt_text)
            print("\n")
            count_tokens(GEMINI_API_KEY, prompt_text)

            print("Token count generated successfully.")

        elif single_img_file is not None:
            # Image
            img_file = os.path.join(img_dir, single_img_file)

            # Type 
            type = object_type.split(', ') if isinstance(object_type, str) else []

            image = get_image(img_file)

            result = []

            print(f"Processing: {single_img_file}")  
            try:
                prompt_text += f'\nThese are the tag categories to which the object belongs, which you can use as an additional reference to narrow down your search domain: "types": {type}'
                # Generate Response
                response_data, time_taken = gen_batch_response(GEMINI_API_KEY, image, type, prompt_text)

                print(f"Response: \n{response_data}")
                print(f"Time taken: {time_taken:.2f} seconds")
                if should_count_tokens:
                    count_tokens(GEMINI_API_KEY, prompt_text)

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
                result.append({
                    "Image": single_img_file,
                    "Types": ", ".join(type),
                    "Title": title,
                    "Description": description,
                    "Tags": tags,
                    "Fields": fields,
                    "Time": time_taken,
                    "Json Response": json_response
                })

            except Exception as e:
                print(f"Error in getting response for {single_img_file}: {e}")
                # Adding an empty result
                result.append({
                    "Image": single_img_file,
                    "Types": ", ".join(type),
                    "Title": "",
                    "Description": "",
                    "Tags": "",
                    "Fields": f"Error: {str(e)}",
                    "Time": 0.0,
                    "Json Response": ""
                })
   
            # Create a new DataFrame with results to ensure proper types
            results_df = pd.DataFrame(result)

            # Format the Time column to have 2 decimal places
            results_df['Time'] = results_df['Time'].apply(lambda x: f"{x:.2f}")

            csv_file_new = get_next_filename(csv_file_dir)

            # Save the results to CSV, replacing the original file
            results_df.to_csv(csv_file_new, index=False)
            
            print(f"Processing complete! Result saved in {csv_file_new}.")
if __name__ == '__main__':
    main()