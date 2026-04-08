
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from Data.raw_data.metadata import generate_metadata


BASE_PATH = r"C:\Users\jiyaa\OneDrive\Desktop\Football_prediction_model\Data\raw_data\football_metadata\data\Premier_League"
LEAGUE = "Premier League"

metadata_df = generate_metadata(BASE_PATH, LEAGUE )

print(metadata_df.head())
print("Metadata shape:", metadata_df.shape)

metadata_df.to_csv(
    "Data/raw_data/premier_league_metadata.csv",
    index=False
)
