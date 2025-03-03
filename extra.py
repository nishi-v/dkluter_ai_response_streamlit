import pandas as pd
from collections import defaultdict

def batch_images(csv_file, batch_size=2):
    # Read the CSV file, skipping the header
    df = pd.read_csv(csv_file, header=0)  # Assuming the first row is the header
    
    # Dictionary to store image names and their corresponding types
    image_dict = defaultdict(list)

    # Populate dictionary with image names as keys and list of types as values
    for _, row in df.iterrows():
        image_name, types = row[0], row[1]
        if image_name != "Image":  # Avoid header inclusion
            image_dict[image_name].append(types)  # Ensure types are split correctly

    # Convert dictionary to a list of tuples
    image_list = list(image_dict.items())

    # Create batches of images
    batches = [image_list[i:i + batch_size] for i in range(0, len(image_list), batch_size)]

    return batches

# Example usage
batches = batch_images("/home/nishita/Downloads/futuresoft/Dkluter_ai_response/csv_files/Items_List (copy).csv")
print(batches)