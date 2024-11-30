import re
import pandas as pd

def parse_log_file(file_path):
    # Define a regex pattern to extract data from relevant lines
    pattern = re.compile(
        r"\[(?P<timestamp>[\d\- :,.]+) INFO (?P<client>\w+)\]:\s+"
        r"(?P<round>\d+)\s+(?P<time>\d+\.\d+)\s+(?P<train_loss>\d+\.\d+)\s+"
        r"(?P<rmse>\d+\.\d+)\s+(?P<val_loss>\d+\.\d+)\s+(?P<val_rmse>\d+\.\d+)"
    )

    # Initialize a list to store extracted data
    data = []

    # Read the log file line by line
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Extract data into a dictionary
                data.append(match.groupdict())
    
    # Convert list of dictionaries to a Pandas DataFrame for easy manipulation
    if data:
        df = pd.DataFrame(data)
        # Convert numeric columns from strings to appropriate data types
        numeric_columns = ['round', 'time', 'train_loss', 'rmse', 'val_loss', 'val_rmse']
        df[numeric_columns] = df[numeric_columns].astype(float)
        return df
    else:
        return None

# Path to the log file
file_path = "/home/ankith/github/smart-home-fed-learning/src/output_final/result_Client4_2024-11-29-18:17:31.txt"

# Parse the file and display the data
df = parse_log_file(file_path)
if df is not None:
    print(df)
    # Save to CSV for further use
    df.to_csv("extracted_data_4.csv", index=False)
    print("Data has been saved to 'extracted_data.csv'.")
else:
    print("No data found in the log file.")
