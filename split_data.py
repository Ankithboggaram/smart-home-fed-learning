import pandas as pd
import os

def split_csv_round_robin(input_csv, num_splits=5, output_folder='data/split_data'):
    
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False, dtype={0:str, 27:str})
    
    dfs = [pd.DataFrame() for _ in range(num_splits)]
    
    for i in range(len(df)):
        dfs[i % num_splits] = pd.concat([dfs[i % num_splits], df.iloc[[i]]], ignore_index=True)
    
    for i in range(num_splits):
        output_file = os.path.join(output_folder, f"home_{i + 1}.csv")
        dfs[i].to_csv(output_file, index=False)
        print(f"Saved {output_file}")

split_csv_round_robin('data/HomeC.csv', num_splits=5, output_folder='data/split_data')
