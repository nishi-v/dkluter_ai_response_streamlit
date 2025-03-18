import streamlit as st
import pandas as pd
import base64
import os
import subprocess
from io import BytesIO
from PIL import Image
import csv
import sys
from streamlit.runtime.uploaded_file_manager import UploadedFile 
import re
import zipfile
import shutil

# Directories for saving images and CSV
IMAGE_SAVE_DIR = "asset_images"
CSV_SAVE_DIR = "csv_files"

# Ensure the directories exist
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(CSV_SAVE_DIR, exist_ok=True)

# Sample CSV Format
sample_data = pd.DataFrame({
    "Image": ["Image1.png", "Image2.jpg"],
    "Types": ["Type1", "Type2"],
    "Title": ["", ""],
    "Description": ["", ""],
    "Tags": ["", ""],
    "Fields": ["", ""]
})

# Function to convert an uploaded image to Base64 (for preview in Streamlit)
def image_to_base64(uploaded_file):
    image = Image.open(uploaded_file)
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Convert all images to PNG for consistency
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# Function to save uploaded images
def save_uploaded_image(uploaded_file:UploadedFile)->str:
    file_path = os.path.join(IMAGE_SAVE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Save image as file
    return file_path  # Return saved file path

# Function to save uploaded CSV
def save_uploaded_csv(uploaded_file:UploadedFile)->str:
    csv_file_path = os.path.join(CSV_SAVE_DIR, uploaded_file.name)
    with open(csv_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return csv_file_path

# Function to extract images from a zip file
def extract_images_from_zip(zip_file:UploadedFile)->list:
    # Clean the image directory first
    shutil.rmtree(IMAGE_SAVE_DIR, ignore_errors=True)
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    
    extracted_files = []
    
    with zipfile.ZipFile(BytesIO(zip_file.getbuffer())) as z:
        for filename in z.namelist():
            # Only extract image files
            lower_filename = filename.lower()
            if lower_filename.endswith(('.jpg', '.jpeg', '.png')):
                # Extract only files, not directories
                if not filename.endswith('/'):
                    z.extract(filename, IMAGE_SAVE_DIR)
                    # Strip any directory structure and just get the basename
                    base_filename = os.path.basename(filename)
                    # If needed, move files from nested directories to the root image directory
                    if filename != base_filename:
                        extracted_path = os.path.join(IMAGE_SAVE_DIR, filename)
                        target_path = os.path.join(IMAGE_SAVE_DIR, base_filename)
                        # Create parent directories if they don't exist
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        # Move the file if it's not already there
                        if os.path.exists(extracted_path) and extracted_path != target_path:
                            shutil.move(extracted_path, target_path)
                    
                    extracted_files.append(base_filename)
    
    return extracted_files

def get_csv_download_link(df: pd.DataFrame, file_path: str)-> str:
    """
    Generates a download link for the processed CSV file.
    """
    csv = df.to_csv(index=False)  # Convert DataFrame to CSV string
    b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV as Base64
    filename = os.path.basename(file_path)
    
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Output CSV</a>'

def get_sample_csv_download_link(df):
    csv = df.to_csv(index=False)  # Convert DataFrame to CSV
    b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV as Base64
    return f'<a href="data:file/csv;base64,{b64}" download="sample_template.csv">📥 Download Example CSV</a>'

# Function to display results from CSV
def display_results(csv_path:str)->None:
    if os.path.exists(csv_path):
        # Read the CSV file
        updated_df = pd.read_csv(csv_path)
        
        # Extract summary row (last row) and remove it from the main dataframe
        if "SUMMARY" in updated_df["Image"].values:
            summary_row = updated_df[updated_df["Image"] == "SUMMARY"].iloc[0].to_dict()
            results_df = updated_df[updated_df["Image"] != "SUMMARY"].copy()
        else:
            summary_row = None
            results_df = updated_df.copy()
        
        # Display results
        st.subheader("Analysis Results")
        
        # Display each image with its analysis
        for i, (index, row) in enumerate(results_df.iterrows()):
            with st.container():
                st.markdown(f"### Image {i + 1}")
                
                # Create columns for image, content, and tags
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    # Display image
                    image_path = os.path.join(IMAGE_SAVE_DIR, row['Image'])
                    if os.path.exists(image_path):
                        st.image(image_path, width=200)
                    else:
                        st.warning(f"Image not found: {row['Image']}")
                
                with col2:
                    # Display title and description
                    st.markdown(f"**Title:** {row['Title']}")
                    st.markdown(f"**Description:**")
                    st.markdown(f"{row['Description']}")
                
                with col3:
                    # Display tags as a list
                    st.markdown("**Tags:**")
                    if isinstance(row['Tags'], str) and row['Tags']:
                        tags = row['Tags'].split(',')
                        for tag in tags:
                            st.markdown(f"- {tag.strip()}")
                    
                    # Display fields
                    if 'Fields' in row and isinstance(row['Fields'], str) and row['Fields']:
                        st.markdown("**Fields:**")
                        
                        # Regular expression to match ", " ONLY when followed by another field pattern
                        field_pattern = r', (?=[A-Za-z\s]+\(TEXT\)|[A-Za-z\s]+\(NUMBER\)|[A-Za-z\s]+\(DATE\))'
                        
                        fields = re.split(field_pattern, row['Fields'])  # Split only at correct places
                        
                        for field in fields:
                            st.markdown(f"- {field.strip()}")  # Display each field on a new line
                
                st.markdown("---")
        
        # Display summary separately
        if summary_row:
            st.subheader("Summary")
            summary_container = st.container()
            with summary_container:
                cols = st.columns(3)
                
                with cols[0]:
                    st.metric("Total Images", summary_row.get("Types", "").split(": ")[-1])
                    st.metric("Successfully Processed", summary_row.get("Title", "").split(": ")[-1])
                    st.metric("Failed", summary_row.get("Description", "").split(": ")[-1])
                
                with cols[1]:
                    # Parse time from Fields or Tags
                    total_time = summary_row.get("Tags", "")
                    if "Total:" in total_time:
                        full_time = total_time.split("Total:")[-1].strip()
                        st.metric("Total Processing Time", full_time)
                
                with cols[2]:
                    # Parse time from Fields or Tags
                    time_text = summary_row.get("Fields", "")
                    if "Average:" in time_text:
                        avg_time = time_text.split("Average:")[-1].strip()
                        st.metric("Average Processing Time per Image", avg_time)
        
        # Show raw data in expandable section
        with st.expander("Show CSV Output Data"):
            st.dataframe(updated_df)

        # 🔹 **Add Download Button**
        st.markdown(get_csv_download_link(updated_df, csv_path), unsafe_allow_html=True)
        
    else:
        st.error("CSV file not found!")

# Function to validate if image files in CSV exist in the extracted files
def validate_csv_images(csv_path:str, extracted_files:list)->list:
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            image_column = df["Image"]
            
            missing_images = []
            for image in image_column:
                if image != "SUMMARY" and image not in extracted_files:
                    missing_images.append(image)
            
            return missing_images
        except Exception as e:
            return ["Error reading CSV: " + str(e)]
    return ["CSV file not found"]

# Streamlit UI
st.title("Image Analysis Tool")

# Choose workflow option
workflow = st.radio("Select Workflow", ["Upload Images", "Upload CSV"])

if workflow == "Upload Images":
    # Textbox for CSV filename
    csv_filename = st.text_input("Enter a CSV filename (without extension)", value="output")

    st.info("Upload single or multiple images!")

    # Upload multiple image files
    uploaded_files = st.file_uploader(
        label="Choose image files ('jpg','jpeg','png')", 
        type=['jpg', 'jpeg', 'png'], 
        accept_multiple_files=True
    )

    # Initialize session state for types if it doesn't exist
    if "types" not in st.session_state:
        st.session_state.types = []

    # Ensure types list is the correct length
    if uploaded_files:
        # Resize the types list if needed
        if len(st.session_state.types) < len(uploaded_files):
            # Extend the list with empty strings if there are new images
            st.session_state.types.extend([""] * (len(uploaded_files) - len(st.session_state.types)))
        elif len(st.session_state.types) > len(uploaded_files):
            # Trim the list if images were removed
            st.session_state.types = st.session_state.types[:len(uploaded_files)]
        
        # Display image previews and type inputs
        for i, file in enumerate(uploaded_files):
            col1, col2 = st.columns([2, 1])
            
            # Display image in the first column
            with col1:
                st.image(file, width=100, caption=f"Image {i+1}")
            
            # Display type input in the second column
            with col2:
                st.session_state.types[i] = st.text_input(
                    f"Type for image {i+1}", 
                    value=st.session_state.types[i],
                    key=f"type_{i+1}"
                )

        # Auto-save files and prepare CSV data when files are uploaded
        csv_data = []
        for i, file in enumerate(uploaded_files):
            image_path = save_uploaded_image(file)  # Save each image
            csv_data.append({
                "Image": file.name,
                "Types": st.session_state.types[i],
                "Title": "",
                "Description": "",
                "Tags": "",
                "Fields": ""
            })
                
        # Generate CSV filename
        csv_relative_name = f"{csv_filename}.csv"  
        csv_full_path = os.path.join(CSV_SAVE_DIR, csv_relative_name)

        # Define all field names
        fieldnames = ["Image", "Types", "Title", "Description", "Tags", "Fields"]

        # Write CSV file
        with open(csv_full_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        # st.success(f"Images saved in '{IMAGE_SAVE_DIR}/' and CSV saved as '{csv_full_path}'")

        # Store the relative CSV name for running the CLI
        st.session_state["csv_name"] = csv_relative_name  
        st.session_state["csv_full_path"] = csv_full_path

    # Button to execute CLI command
    if "csv_name" in st.session_state:
        if st.button("Generate AI Response"):
            with st.spinner("Generating Response..."):
                command = [sys.executable, "main.py", "-f", st.session_state["csv_name"]]
                
                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
                    st.success("Command executed successfully!")
                    
                    # Display the results
                    display_results(st.session_state["csv_full_path"])
                    
                except subprocess.TimeoutExpired:
                    st.error("The command timed out! Please try again with a smaller dataset or check the system performance.")
                except subprocess.CalledProcessError as e:
                    st.error("Error while executing command.")
                    st.text_area("Error Message", e.stderr)

else:
    # CSV Upload workflow
    st.subheader("Upload CSV File")

    st.markdown("### Example CSV Format:")
    st.dataframe(sample_data)  # Display example CSV structure
    st.markdown(get_sample_csv_download_link(sample_data), unsafe_allow_html=True)
    
    # Upload CSV file
    uploaded_csv = st.file_uploader("Choose a CSV file", type=['csv'])

    st.info("Create a zip file of all the images mentioned in CSV file and upload it here.")

    # Upload ZIP file containing images
    uploaded_zip = st.file_uploader("Upload ZIP file containing images", type=['zip'])
    
     # Variables to track state
    csv_path = None
    extracted_files = []
    csv_data = None

    if uploaded_csv is not None:
        # Save the uploaded CSV
        csv_path = save_uploaded_csv(uploaded_csv)

        # Read the CSV data
        try:
            csv_data = pd.read_csv(csv_path)
            # Initialize types in session state if needed
            if "csv_types" not in st.session_state:
                st.session_state.csv_types = csv_data["Types"].tolist()
            elif len(st.session_state.csv_types) != len(csv_data):
                st.session_state.csv_types = csv_data["Types"].tolist()
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
        
    if uploaded_zip is not None:
        # Extract images from ZIP
        with st.spinner("Extracting images from ZIP file..."):
            extracted_files = extract_images_from_zip(uploaded_zip)
        
        st.success(f"Successfully extracted {len(extracted_files)} images.")

        # # Show the first few extracted images as preview
        # if extracted_files:
        #     st.subheader("Sample Extracted Images")
        #     preview_count = min(5, len(extracted_files))
        #     cols = st.columns(preview_count)
            
        #     for i in range(preview_count):
        #         with cols[i]:
        #             img_path = os.path.join(IMAGE_SAVE_DIR, extracted_files[i])
        #             if os.path.exists(img_path):
        #                 st.image(img_path, width=100, caption=extracted_files[i])

    # Check if both CSV and ZIP have been uploaded
    if csv_path and extracted_files and csv_data is not None:
        # Validate that all images in the CSV exist in the extracted files
        missing_images = validate_csv_images(csv_path, extracted_files)

        if missing_images:
            st.warning("Warning: The following images referenced in the CSV were not found in the ZIP file:")
            for img in missing_images[:10]:  # Show first 10 missing images
                st.write(f"- {img}")
            if len(missing_images) > 10:
                st.write(f"... and {len(missing_images) - 10} more.")

         # Display images with type inputs side by side (similar to Upload Images workflow)
        st.subheader("Preview Images and Types")

        # Filter out summary row if it exists
        display_data = csv_data[csv_data["Image"] != "SUMMARY"] if "SUMMARY" in csv_data["Image"].values else csv_data

         # Display each image with its type input
        for i, (_, row) in enumerate(display_data.iterrows()):
            image_name = row['Image']
            image_path = os.path.join(IMAGE_SAVE_DIR, image_name)
            
            # Create two columns layout for each image
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display image if it exists
                if os.path.exists(image_path):
                    st.image(image_path, width=100, caption=f"Image: {i+1}")
                else:
                    st.warning(f"Image not found: {image_name}")
            
            with col2:
                # Allow editing the type for each image
                new_type = st.text_input(
                    f"Type for image {i+1}", 
                    value=row['Types'],
                    key=f"type_{i+1}"
                )

                # Auto-update the type value in session state
                st.session_state.csv_types[i] = new_type
                
                # Update the CSV data immediately when type changes
                if new_type != row['Types']:
                    csv_data.at[display_data.index[i], 'Types'] = new_type
                    # Save the updated CSV
                    csv_data.to_csv(csv_path, index=False)
        # Auto-save updated types - this happens automatically when the types are changed above
        st.info("Types are automatically updated as you change them")

        # Extract just the filename for the CLI command
        csv_filename = os.path.basename(csv_path)

        # Button to process the uploaded CSV
        if st.button("Generate AI Response"):
            with st.spinner("Generating Response..."):
                command = [sys.executable, "main.py", "-f", csv_filename]
                
                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
                    st.success("Command executed successfully!")
                    
                    # Display the results
                    display_results(csv_path)
                    
                except subprocess.TimeoutExpired:
                    st.error("The command timed out! Please try again with a smaller dataset or check the system performance.")
                except subprocess.CalledProcessError as e:
                    st.error("Error while executing command.")
                    st.text_area("Error Message", e.stderr)
