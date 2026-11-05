# Bank of Thailand Monetary Policy Report (MPR) web download script

import requests
from pathlib import Path
from datetime import datetime

# Initial setup
BASE_URL = (
    "https://www.bot.or.th/content/dam/bot/documents/en/"
    "our-roles/monetary-policy/mpc-publication/"
    "monetary-policy-report/"
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "documents"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2013
START_QUARTER = 1
CURRENT_YEAR = datetime.now().year

# Create a resuable requests session with a custom User-Agent
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})


# Download MPR reports
print("Beginning download of Bank of Thailand MPR reports...")

for year in range(START_YEAR, CURRENT_YEAR + 1):
    for quarter in range(1, 5):

        if year > CURRENT_YEAR:
            print(f"Reached future year ({year}). Terminating.")
            raise SystemExit

        filename = f"MPR_{year}_Q{quarter}.pdf"
        filepath = OUTPUT_DIR / filename
        url = BASE_URL + filename

        # Check if already downloaded
        if filepath.exists():
            print(f"FILENAME ALREADY EXISTS: {filename}")
            continue

        print(f"DOWNLOADING: {filename}")

        try:
            response = session.get(
                url,
                stream=True,
                timeout=30
            )

            if response.status_code == 404:
                print(f"FILE NOT FOUND: {filename}")
                print("No further reports available. Terminating.")
                raise SystemExit

            response.raise_for_status()

            # Verify PDF signature
            first_bytes = response.raw.read(5)

            if first_bytes != b"%PDF-":
                print(f"INVALID PDF: {filename}")
                print("Terminating.")
                raise SystemExit

            # Download PDF
            with open(filepath, "wb") as f:
                f.write(first_bytes)

                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)

            print(f"DOWNLOADED: {filename}")

        except requests.RequestException as e:
            print(f"ERROR {filename}: {e}")
            raise SystemExit

print("Finished downloading Bank of Thailand MPR reports.")
