import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from libraries.core import *
from datetime import datetime

def generate_metadata(base_path,league):
    metadata_records=[]

    for season in os.listdir(base_path):
        season_path=os.path.join(base_path,season)

        if not os.path.isdir(season_path):
            continue
        for dataset_type in os.listdir(season_path):
            dataset_path = os.path.join(season_path, dataset_type)

            if not os.path.isdir(dataset_path):
                continue

            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    file_lower = file.lower()
                    file_path = os.path.join(root, file)

                    try:
                        if file_lower.endswith(".csv"):
                            df = pd.read_csv(file_path, encoding="latin1")

                        elif file_lower.endswith(".csv.gz"):
                            df = pd.read_csv(file_path, compression="gzip")

                        elif file_lower.endswith(".json"):
                            df = pd.read_json(file_path)

                        else:
                            continue

                        # Detect Gameweek if present
                        gameweek = None
                        for part in root.split(os.sep):
                            if part.upper().startswith("GW"):
                                gameweek = part
                                break

                        metadata_records.append({
                            "league": league,
                            "season": season,
                            "dataset_type": dataset_type,
                            "gameweek": gameweek,
                            "file_name": file,
                            "rows": df.shape[0],
                            "columns": df.shape[1],
                            "column_names": list(df.columns),
                            "source_path": file_path,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                    except Exception as e:
                        print(f"⚠️ Skipped {file_path} | Reason: {e}")

    return pd.DataFrame(metadata_records)
