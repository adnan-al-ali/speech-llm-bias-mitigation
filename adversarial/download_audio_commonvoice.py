import dotenv
import requests
import os
from tqdm import tqdm

dotenv.load_dotenv()
response = requests.post(
    "https://mozilladatacollective.com/api/datasets/cmndapwry02jnmh07dyo46mot/download",
    headers={
        "Authorization": f"Bearer {os.getenv('MDC_KEY')}",
        "Content-Type": "application/json"
    }
)
response.raise_for_status()

os.makedirs("downloads", exist_ok=True)

download_url = response.json().get("downloadUrl")
if download_url:
    with requests.get(download_url, stream=True) as download_response:
        download_response.raise_for_status()
        total_size = int(download_response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024 * 1024  # 1 GiB chunks

        with open("downloads/CV_English.tar.gz", "wb") as f, tqdm(
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading CV_English.tar.gz",
        ) as pbar:
            for chunk in download_response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
else:
    print("Failed to get download URL.")