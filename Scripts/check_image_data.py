import pandas as pd
import os

# Check if file exists
image_data_path = "data/raw/images/archive/Data_Entry_2017.csv"
if os.path.exists(image_data_path):
    print(f"File exists: {image_data_path}")
    df = pd.read_csv(image_data_path)
    print(f"Columns: {df.columns.tolist()}")
    print(f"First 2 rows:\n{df.head(2)}")
else:
    print(f"File does not exist: {image_data_path}")
    
    # List files in directory
    image_dir = "data/raw/images/archive"
    if os.path.exists(image_dir):
        print(f"Files in {image_dir}:")
        for f in os.listdir(image_dir):
            print(f"  - {f}")
    else:
        print(f"Directory does not exist: {image_dir}") 