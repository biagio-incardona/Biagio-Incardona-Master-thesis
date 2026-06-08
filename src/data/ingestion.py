import os
import shutil
import logging
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/Predizioni-Epidemiologiche-Italia/Influcast.git"
RAW_DATA_DIR = "data/raw"
TEMP_REPO_DIR = "data/influcast_repo"

def ingest_ili_data():
    """
    Ingest ILI data by cloning the Influcast repository and copying relevant files.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Clone or update the repository
    if os.path.exists(TEMP_REPO_DIR):
        logger.info(f"Updating existing repository in {TEMP_REPO_DIR}...")
        try:
            subprocess.run(["git", "-C", TEMP_REPO_DIR, "pull"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update repository: {e}")
            # If update fails, we can still try to use what's there
    else:
        logger.info(f"Cloning repository {REPO_URL} into {TEMP_REPO_DIR}...")
        try:
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, TEMP_REPO_DIR], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            return

    # Path to ILI data in the repo
    ili_root = os.path.join(TEMP_REPO_DIR, "sorveglianza", "ILI")
    if not os.path.exists(ili_root):
        logger.error(f"ILI data root not found in repo at {ili_root}")
        return

    regions = ["italia", "Lombardia", "Lazio", "Sicilia", "Veneto", "Piemonte", "Abruzzo"]
    
    # Walk through seasons
    for season in os.listdir(ili_root):
        season_path = os.path.join(ili_root, season)
        if not os.path.isdir(season_path):
            continue
            
        target_season_dir = os.path.join(RAW_DATA_DIR, season)
        os.makedirs(target_season_dir, exist_ok=True)
        
        # Check if 'latest' folder exists in the season
        latest_dir = os.path.join(season_path, "latest")
        if os.path.exists(latest_dir):
            for region in regions:
                filename = f"{region}-latest-ILI.csv"
                src_file = os.path.join(latest_dir, filename)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, os.path.join(target_season_dir, filename))
                    logger.info(f"Copied {filename} for season {season}")
        else:
            # Find the file with the highest week number for each region
            for region in regions:
                region_lower = region.lower()
                files = [f for f in os.listdir(season_path) if f.lower().startswith(region_lower) and f.endswith("-ILI.csv")]
                if files:
                    files.sort()
                    latest_filename = files[-1]
                    src_file = os.path.join(season_path, latest_filename)
                    shutil.copy2(src_file, os.path.join(target_season_dir, latest_filename))
                    logger.info(f"Copied {latest_filename} for season {season}")
                else:
                    logger.warning(f"No ILI data found for {region} in season {season}")

    # Optionally remove the cloned repo to save space, but keeping it for now as a local cache
    # shutil.rmtree(TEMP_REPO_DIR)

if __name__ == "__main__":
    ingest_ili_data()
