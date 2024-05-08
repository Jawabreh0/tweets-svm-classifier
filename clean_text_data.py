import pandas as pd
import re

def clean_text_data(input_file):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(input_file)
    
    # Define a function to clean text
    def clean_text(text):
        text = text.replace("RT", "")  # Remove the word "RT"
        text = re.sub(r"@\S+", "", text)  # Remove '@' and following characters until a whitespace
        text = re.sub(r"#\S+", "", text)  # Remove '#' and following characters until a whitespace
        text = re.sub(r"\d+", "", text)  # Remove all numbers
        return text.strip()
    
    # Apply the cleaning function to each text column
    for column in data.columns:
        # Check if the column data type is string
        if pd.api.types.is_string_dtype(data[column]):
            data[column] = data[column].apply(clean_text)
    
    # Save the cleaned data back to the original CSV file
    data.to_csv(input_file, index=False)
    print(f"File '{input_file}' has been cleaned and overwritten.")

# Specify the file name
file_name = "normal.csv"
clean_text_data(file_name)
