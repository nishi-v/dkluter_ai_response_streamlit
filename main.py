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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

# Create a thread pool for CPU-bound tasks like image loading
thread_pool = ThreadPoolExecutor(max_workers=10)

async def get_image(img_file: str) -> Image.Image:
    loop = asyncio.get_running_loop()
    try:
        img = await loop.run_in_executor(thread_pool, lambda: Image.open(img_file))
        return img
    except Exception as e:
        print(f"Error fetching {img_file}: {e}")
        return None  # type: ignore

# def count_tokens(api_key:str, prompt_text:str) -> Any:
#     client = genai.Client(api_key = api_key)

#     token_count = client.models.count_tokens(
#         model='gemini-2.0-flash',
#         contents=prompt_text,
#     )
#     print(f"Prompt tokens: {token_count}")

#     return token_count

async def gen_response(api: str, img: Image.Image, types: list, prompt_text: str) -> tuple[dict, float, Any, Any, str]:
    """
    Generate AI response using a thread pool to handle the synchronous API call.
    This allows multiple API calls to run concurrently without blocking the event loop.
    """
    client = genai.Client(api_key=api)
    model_id = "gemini-2.0-flash"

    google_search_tool = Tool(
        google_search=GoogleSearch(),
    )

    start_time = time.time()
    loop = asyncio.get_running_loop()

    # token_count = client.models.count_tokens(
    #     model='gemini-2.0-flash',
    #     contents=prompt_text,
    # )
    # total_tokens = token_count.total_tokens

    # print(f"Prompt tokens: {token_count}")
    try:
        response = await loop.run_in_executor(
            thread_pool,
            lambda: client.models.generate_content(
                model=model_id,
                contents=[img, prompt_text],
                config=GenerateContentConfig(
                    temperature=0.0,
                    seed=42,
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                )
            )
        )
        end = time.time() - start_time
        # print(f"Raw Response: \n\n{response}\n\n")
        if response.usage_metadata:
            output_token_count = response.usage_metadata.candidates_token_count
            input_token_count = response.usage_metadata.prompt_token_count
        else:
            # Provide default values or handle the case where metadata is not available
            output_token_count = 0
            input_token_count = 0

        if response.candidates and response.candidates[0].grounding_metadata:
            grounding_chunks = response.candidates[0].grounding_metadata.grounding_chunks
        else:
            grounding_chunks = None

        if grounding_chunks is None:
            search_tool_used = "No"
        elif grounding_chunks is not None:
            search_tool_used = "Yes"
        # print(f"Grounding chunks: {grounding_chunks}\n")
        print(f"Grounding google Search tool used or not: {search_tool_used}")


        if response.text:
            raw_response = response.text.strip()
            # Remove markdown code block markers (```json ... ```)
            cleaned_json = re.sub(r"^```json\n|\n```$", "", raw_response)
            try:
                return json.loads(cleaned_json), end, input_token_count, output_token_count, search_tool_used
            except json.JSONDecodeError:
                print(f"Error parsing JSON for image")
                return {"Data": {"title": "", "description": "", "tags": [], "fields": []}}, round(end, 2), input_token_count, output_token_count, search_tool_used
        else:
            return {"Data": {"title": "", "description": "", "tags": [], "fields": []}}, round(end, 2), input_token_count, output_token_count, search_tool_used
    except Exception as e:
        print(f"Error generating response: {e}")
        return {"Data": {"title": "", "description": "", "tags": [], "fields": []}}, round(time.time() - start_time, 2), input_token_count, output_token_count, search_tool_used

def get_next_filename(dir: str) -> str:
    existing_files = glob.glob(os.path.join(dir, "output_*.csv"))  
    next_num = len(existing_files) + 1  
    return os.path.join(dir, f"output_{next_num:05d}.csv")

async def process_img(img_name: str, img_dir: str, types: List[str], prompt_text: str, api_key: str) -> Dict[str, Any]:
    """Process a single image with the AI model"""
    img_file = os.path.join(img_dir, img_name)
    image = await get_image(img_file)
    if image is None:
        return {
            "Image": img_name,
            "Types": ", ".join(types),
            "Title": "",
            "Description": "",
            "Tags": "",
            "Fields": f"Error: Could not load image",
            "Time": 0.0,
            "Input Token Count": "",
            "Output Token Count": "",
            "Search Tool Used": "",
            "Json Response": ""
        }

    try:
        full_prompt = prompt_text + f'\nThese are the tag categories to which the object belongs, which you can use as an additional reference to narrow down your search domain: "types": {types}'

        # Generate Response
        response_data, time_taken, input_tokens, output_tokens, search_tool = await gen_response(api_key, image, types, full_prompt)

        print(f"Response: {response_data}\n")

        print(f"Input Tokens: {input_tokens}")
        print(f"Output Tokens: {output_tokens}")

        print(f"Processed: {img_name} in {time_taken:.2f} seconds")
        # if count:
        #     # Run token counting in thread pool since it's synchronous
        #     await asyncio.get_running_loop().run_in_executor(
        #         thread_pool, lambda: count_tokens(api_key, full_prompt))

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
                field_value_str = ', '.join([''.join(v) if isinstance(v, list) else str(v) for v in field_values])
            else:
                field_value_str = str(field_values)  # Handle non-list cases safely

            fields_data.append(f"{field_name} ({field_type}): {field_value_str}")

        fields = ', '.join(fields_data)

        # Store the complete JSON response
        json_response = json.dumps(response_data)

        # Return the result
        return {
            "Image": img_name,
            "Types": ", ".join(types),
            "Title": title,
            "Description": description,
            "Tags": tags,
            "Fields": fields,
            "Time": time_taken,
            "Input Token Count": input_tokens,
            "Output Token Count": output_tokens,
            "Search Tool Used": search_tool,
            "Json Response": json_response
        }

    except Exception as e:
        print(f"Error in getting response for {img_name}: {e}")
        return {
            "Image": img_name,
            "Types": ", ".join(types),
            "Title": "",
            "Description": "",
            "Tags": "",
            "Fields": f"Error: {str(e)}",
            "Time": 0.0,
            "Input Token Count": "",
            "Output Token Count": "",
            "Search Tool Used": "",
            "Json Response": ""
        }

async def process_csv_file(csv_file_path: str, img_dir: str, api_key: str, prompt_text: str, max_concurrent: int = 5) -> None:
    """Process all images from a CSV file with controlled concurrency"""
    print(f"Generating Responses...")

    df = pd.read_csv(csv_file_path)
    df = df[df['Image'] != 'SUMMARY']

    results = []
    total_time = 0
    input_total_tokens_all = 0
    output_total_tokens_all = 0
    successful_images = 0
    total_images = len(df)
    search_tool_total_usage = 0

    # print(max_concurrent)
    
    # Create semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_process(img_name: str, types: List[str]):
        """Process an image with a semaphore to limit concurrency"""
        async with semaphore:
            return await process_img(img_name, img_dir, types, prompt_text, api_key)

    # Create tasks for all images
    tasks = []
    for index, row in df.iterrows():
        img_name = row['Image']
        if img_name == 'SUMMARY':
            continue
            
        types = row['Types'].split(', ') if isinstance(row['Types'], str) else []
        print(f"Queueing {index+1}: {img_name}") #type:ignore
        
        task = limited_process(img_name, types)
        tasks.append(task)
    
    # Run all tasks concurrently with the semaphore controlling max concurrency
    results = await asyncio.gather(*tasks)
        
    # Update metrics
    for result in results:
        time_taken = float(result.get("Time", 0))
        input_token_count = result.get("Input Token Count", 0)
        output_token_count = result.get("Output Token Count", 0)
        search_tool_usage = result.get("Search Tool Used", 0)
        if time_taken > 0 and result.get("Title", ""):
            total_time += time_taken
            successful_images += 1

        if input_token_count:
            input_total_tokens_all += input_token_count
        if output_token_count:
            output_total_tokens_all += output_token_count
        
        if search_tool_usage == "Yes":
            search_tool_total_usage += 1
            
    
    # Calculating average time
    avg_time = total_time / successful_images if successful_images > 0 else 0
    avg_input_tokens = int(input_total_tokens_all / successful_images) if successful_images > 0 else 0
    avg_output_tokens = int(output_total_tokens_all / successful_images) if successful_images > 0 else 0
    
    # Create a new DataFrame with results to ensure proper types
    results_df = pd.DataFrame(results)
    
    # Format the Time column to have 2 decimal places
    results_df['Time'] = results_df['Time'].apply(lambda x: f"{x:.2f}")
    
    # Add summary statistics
    summary_df = pd.DataFrame([{
        "Image": "SUMMARY",
        "Types": f"Total Images: {total_images}",
        "Title": f"Successfully Processed: {successful_images}",
        "Description": f"Failed: {total_images - successful_images}",
        "Tags": f"Total: {total_time:.2f}s",
        "Fields": f"Average: {avg_time:.2f}s",
        "Time": f"Total Input Tokens: {int(input_total_tokens_all)}",
        "Input Token Count": f"Average Input Tokens: {int(avg_input_tokens)}",
        "Output Token Count": f"Total Output Tokens: {int(output_total_tokens_all)}",
        "Search Tool Used": f"Average Output Tokens: {int(avg_output_tokens)}",
        "Json Response": f"Total Search Tool Usage: {int(search_tool_total_usage)}"
    }])
    
    # Combine the results with the summary
    results_df = pd.concat([results_df, summary_df], ignore_index=True)
    
    # Save the results to CSV
    results_df.to_csv(csv_file_path, index=False)
    
    # Print summary statistics
    print(f"\nSUMMARY:")
    print(f"Total Images: {total_images}")
    print(f"Successfully Processed: {successful_images}")
    print(f"Failed: {total_images - successful_images}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Image: {avg_time:.2f} seconds")
    print(f"Total Input Tokens: {int(input_total_tokens_all)}")
    print(f"Average Input Tokens per Image: {int(avg_input_tokens)}")
    print(f"Total Output Tokens: {int(output_total_tokens_all)}")
    print(f"Average Output Tokens per Image: {int(avg_output_tokens)}")
    print(f"Total Search Tool Usage: {int(search_tool_total_usage)}")
    
    print(f"Processing complete! Results saved in {csv_file_path}.")

async def process_single_img(single_img_file: str, object_type: str, img_dir: str, api_key: str, prompt_text: str, csv_file_dir: str)-> None:
    # Image
    img_file = os.path.join(img_dir, single_img_file)
    
    # Type 
    types = object_type.split(', ') if isinstance(object_type, str) else []
    
    print(f"Processing: {single_img_file}")
    
    result = await process_img(single_img_file, img_dir, types, prompt_text, api_key)
    
    # Create a new DataFrame with results
    results_df = pd.DataFrame([result])
    
    # Format the Time column to have 2 decimal places
    results_df['Time'] = results_df['Time'].apply(lambda x: f"{x:.2f}")
    
    csv_file_new = get_next_filename(csv_file_dir)
    
    # Save the results to CSV
    results_df.to_csv(csv_file_new, index=False)
    
    print(f"Processing complete! Result saved in {csv_file_new}.")

async def main_async():
    start_final = time.time()
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-f", "--file", required=False, help="Name of the CSV file")
    # parser.add_argument("-t", "--token", required=False, action="store_true", help="Include it to count tokens")
    parser.add_argument("-T", "--type", required=False, help="Define types if only count token is required")
    parser.add_argument("-i", "--img", required=False, help="Name of single image for response generation")
    parser.add_argument("-c", "--concurrent", type=int, default=5, help="Maximum number of concurrent requests")
    
    args = parser.parse_args()
    
    csv_file = args.file
    # should_count_tokens = args.token  # Will be true if -t is passed, otherwise false.
    object_type = args.type  # Give type of object, if only you want token count to be generated. No response will be generated.
    single_img_file = args.img  # Name of single image you want to generate response for.
    max_concurrent = args.concurrent  # Maximum number of concurrent requests
    
    if csv_file is not None:
        print(f"Received csv file: {csv_file}")
        # if should_count_tokens:
        #     print("Token counts will be generated.")
        # else:
        #     print("Token count will not be generated.")
    elif csv_file is None and object_type is not None:
        if single_img_file is None:
            print(f"Only token count will be generated.")
        elif single_img_file is not None:
            print(f"Received image: {single_img_file}")
            # if should_count_tokens:
            #     print("Token count will be generated.")
            # else:
            #     print("Token count will not be generated.")
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

    # Image file directory
    img_dir :str = os.path.join(dir, 'asset_images')

    # CSV file directory
    csv_file_dir = os.path.join(dir, "csv_files")

    # Get API URL from environment variables
    GEMINI_API_KEY :str = os.environ["GEMINI_API_KEY"]

    if csv_file is not None:
        csv_file_path = os.path.join(csv_file_dir, csv_file)
        await process_csv_file(csv_file_path, img_dir, GEMINI_API_KEY, prompt_text, max_concurrent)
    
    elif csv_file is None and object_type is not None:
        if single_img_file is None:
            prompt_text += f'\nThese are the tag categories to which the object belongs, which you can use as an additional reference to narrow down your search domain: "types": {object_type}' 
            print(prompt_text)
            print("\n")
            # count_tokens(GEMINI_API_KEY, prompt_text)

            # print("Token count generated successfully.")

        elif single_img_file is not None:
            await process_single_img(single_img_file, object_type, img_dir, GEMINI_API_KEY, prompt_text, csv_file_dir)
    
    end_final = time.time() - start_final
    print(f"Total time taken in processing: {end_final}")

def main():
    asyncio.run(main_async())

if __name__ == '__main__':
    main()