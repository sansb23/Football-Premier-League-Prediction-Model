import os
import pandas as pd

def load_raw_data(base_paths):
    df_list = []

    for folder in base_paths:
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()

                if file_lower.endswith(".csv"):
                    temp_df = pd.read_csv(file_path, encoding="latin1")

                elif file_lower.endswith(".csv.gz"):
                    temp_df = pd.read_csv(file_path, compression="gzip")

                elif file_lower.endswith(".json"):
                    temp_df = pd.read_json(file_path)

                else:
                    continue

                temp_df["source_folder"] = os.path.basename(folder)
                df_list.append(temp_df)

    if len(df_list) == 0:
        raise ValueError("No raw data files found")

    return pd.concat(df_list, ignore_index=True, sort=False)
