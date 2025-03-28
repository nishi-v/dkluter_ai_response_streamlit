# D'Kluter Response Generation using AI

## Description
This project utilizes Gemini-2.0-Flash and Google Search to analyze a given image and its associated types. It generates structured metadata, including a title, description, tags, and detailed fields for detected objects.

Batch responses can also be generated by providing a CSV file with multiple images and their corresponding types.

Some sample images and a CSV file with results are also provided for reference.

## How to Use:
### Step 1: Clone the Repository
```bash
git clone https://github.com/AI-ML-Team-FS/dkluter_ai_response.git
cd dkluter_ai_response/
```

### Step 2: Initialize virtual environment
#### Option 1: Using Poetry
1. Install poetry if not already present:
```bash
pip install poetry 
```

2. Ensure that the virtual environment is created in the root directory:
```bash
poetry config virtualenvs.in-project true
```

3. Initialize virtual environment:
```bash
poetry shell
```

4. Install the required packages:
```bash
poetry install
```


#### Option 2: Using requirements.txt
1. Create a virtual environment:
```bash
python3 -m venv .venv
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

3. Install the required packages using requirements file.
```bash
pip install -r requirements.txt
```

### Step 3: Generate and save Gemini API Key
1. Generate an API Key from Google AI Studio:
- link: [API key from Google AI Studio](https://aistudio.google.com/app/apikey)

2. Save the generated API key in .env file in root directory:

    A. Create .env file in the root folder.

    B. Add API Key in .env file:
    - GEMINI_API_KEY = "AIza...4CpSE"

### Step 4: Save the images in desired folder
Add all images you want to generate the response for in the asset_images folder.

Note: Sample images are present in asset_images folder.


### Step 5: Create a csv file
Create a CSV file in desired format and save it in csv_files folder. 
Ensure the CSV file follows this structure. The columns for Title, Description, Tags, Fields, Time, and JSON Response can be left empty—these will be automatically filled by the script.

| S.No. | Image | Types | Title | Description | Tags | Fields | 
|----------|----------|----------|----------|----------|----------|----------|
| 1 | img_name_1.jpg | type_1_1, type_1_2 | | | | | 
| 2 | img_name_1.jpg | type_2_1, type_2_2 | | | | | 
| 3 | img_name_1.jpg | type_3_1, type_3_2 | | | | | 

Note: 
1. Sample CSV files with responses is present in csv_files
    - For batch processing input csv file -> output csv file.
    
    Example: Items_List.csv, list_2.csv

    - For single image file, input image name and type of object -> output csv file. WIll be saved in above format.

    Example: output_00001.csv


## Step 6: Run the python script
There are 5 arguments that you can pass with the python script-
### A. Batch Response Generation using CSV
1. -f <filename>: Specifies the CSV file for batch response generation.
2. -c <number>: (Optional) Sets the maximum number of concurrent requests during batch processing. The default value is 5.

### B. Single Image Response Generation 
1. -i <image_filename>: Specifies the image file for response generation.
2. -T <type>: Specifies the types/categories associated with the image. This argument must be used along with -i.
- It can also be used alone to get only the token count, without generating a response.

<!-- ### C. Universal Argument
- Applicable to both batch and single-image processing:
1. -t : Enables token count calculation for the input data (CSV file or image). -->

Usage:
1. Batch Responses Using CSV
- Generate responses for a CSV file (default concurrent tasks = 5):
```bash
python3 main.py -f data.csv
```

<!-- - Generate responses and get the token count for a CSV file:
```bash
python3 main.py -f data.csv -t
``` -->

<!-- - Generate responses and token count for a CSV file with a custom concurrency limit (e.g., 10):
```bash
python3 main.py -f data.csv -t -c 10
``` -->

2. Single Responses 
<!-- - Generate a response for a single image and retrieve the token count:
```bash
python3 main.py -i "img.jpg" -T "type_1, type_2, type_3" -t
``` -->

- Generate a response for a single image: 
```bash
python3 main.py -i "img.jpg" -T "type_1, type_2, type_3"
```

<!-- - Get only the token count (without generating a response):
```bash
python3 main.py -T "type_1, type_2, type_3"
``` -->


Note:
1. If batch input in CSV format is given, then output will be stored in that CSV file only with a SUMMARY row at the end.

| SUMMARY | Total Images: | Successfully Processed: | Failed: | Total Time: | Average Time: | Total Input Tokens: | Average Input Tokens: | Total Output Tokens: | Average Output Tokens | Search Tool Used: |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|

2. If single image and corresponding type is provided then an automatic new output_8.csv file will be saved in csv_files folder with the response.

## General JSON Format of Response for an Input
- Input given by User:
    - image = image.jpg (jpg, jpeg, or png Format)
    - types = "type_1", "type_2"

- Response Structure - JSON Format:
```json
{
    "Data": {
        "title": "Appropriate Title",
        "description": "A brief description providing relevant details about their characteristics, actions, or context while ignoring the background.",
        "tags": [
            "tag_1",
            "tag_2",
            "tag_3"
        ],
        "fields": [
            {
                "field_name": "Field_1",
                "field_type": "NUMBER",
                "field_value": ["12345"]
            },
            {
                "field_name": "Field_2",
                "field_type": "TEXT",
                "field_value": ["Value"]
            }
        ]
    }
}
```
### Explanation of Response Field: 
- Title: The AI-generated title for the image based on detected objects and context.
- Description: A concise AI-generated description summarizing the content of the image.
- Tags: A list of relevant keywords describing the image, starting from broad categories and moving to specific attributes.
- Fields: Structured metadata fields that provide additional details about detected objects.
    - Field Name: Name of the detected field (e.g., "Product ID", "Location").
    - Field Type: Type of data stored in the field (NUMBER, TEXT, DATE, or LOCATION).
    - Field Value: The corresponding value(s) for the field.

### Response Limitations:

| Field | Limitation |
|----------|----------|
| Title | Max 100 characters |
| Description | Max 80 words |
| Tags | Max 50 characters |
| Field Name | Max 50 characters | 
| Field Type | Must be one of: NUMBER, TEXT, DATE, LOCATION |
| Field Value | Max 500 characters |


## Troubleshooting and FAQs

### Common Issues
1. API Key not working
- Ensure the API key is correctly saved in the .env file.
- Verify that the key is active and has the necessary permissions in Google AI Studio.

2. Module Not Found Error
- Ensure dependencies are installed by running:
```bash
poetry install  # If using Poetry
```

or

```bash
pip install -r requirements.txt  # If using requirements.txt
```

3. CSV File Not Updating
- Ensure the CSV file is properly formatted as per the given structure.
- Check if the script has necessary write permissions for the file.
