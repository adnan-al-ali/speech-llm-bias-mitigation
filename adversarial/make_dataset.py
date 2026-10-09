import duckdb
from huggingface_hub import HfFileSystem
import os

duckdb.register_filesystem(HfFileSystem())

os.makedirs("downloads", exist_ok=True)
splits = ["train"]

for split in splits:
    if not os.path.isfile(f"adversarial/downloads/sift_50m_vctk_en_{split}.parquet"):
        table = duckdb.sql(f"SELECT * FROM read_parquet('hf://datasets/amazon-agi/SIFT-50M@~parquet/*/*{split}/*.parquet') WHERE data_source='vctk_en' AND task <> 'controllable_generation'")

        table.to_parquet(f"adversarial/downloads/sift_50m_vctk_en_{split}.parquet")
