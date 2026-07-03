import os
import shutil
import logging
import subprocess
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/Predizioni-Epidemiologiche-Italia/Influcast.git"
RAW_DATA_DIR = "data/raw"
TEMP_REPO_DIR = "data/influcast_repo"
SOURCE_INDEX_FILE = "data/processed/source_files_index.csv"

def get_available_regions(season_path):
    """
    Discover all regions/provinces available in a given season path.
    Looks for files ending in -ILI.csv and extracts the region name.
    """
    regions = set()
    # Check 'latest' folder if it exists
    latest_dir = os.path.join(season_path, "latest")
    if os.path.exists(latest_dir):
        for f in os.listdir(latest_dir):
            if f.endswith("-latest-ILI.csv"):
                regions.add(f.replace("-latest-ILI.csv", ""))
    
    # Check the season folder itself for weekly files
    for f in os.listdir(season_path):
        if f.endswith("-ILI.csv") and not f.endswith("-latest-ILI.csv"):
            # Format is typically region-YYYY_WW-ILI.csv or Region-YYYY_WW-ILI.csv
            # We take everything before the first hyphen
            region = f.split('-')[0]
            if region:
                regions.add(region)
    
    return list(regions)

def ingest_ili_data():
    """
    Ingest ILI data by cloning the Influcast repository and copying relevant files.
    Implements recursive discovery and source tracking.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SOURCE_INDEX_FILE), exist_ok=True)
    
    # Clone or update the repository
    if os.path.exists(TEMP_REPO_DIR):
        logger.info(f"Updating existing repository in {TEMP_REPO_DIR}...")
        try:
            subprocess.run(["git", "-C", TEMP_REPO_DIR, "pull"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update repository: {e}")
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

    source_index = []
    
    # Walk through seasons
    seasons = sorted([s for s in os.listdir(ili_root) if os.path.isdir(os.path.join(ili_root, s))])
    
    for season in seasons:
        season_path = os.path.join(ili_root, season)
        target_season_dir = os.path.join(RAW_DATA_DIR, season)
        os.makedirs(target_season_dir, exist_ok=True)
        
        # Discover all regions available in this season
        regions = get_available_regions(season_path)
        
        for region in regions:
            # 1. Try 'latest' folder first
            latest_dir = os.path.join(season_path, "latest")
            latest_filename = f"{region}-latest-ILI.csv"
            src_file_latest = os.path.join(latest_dir, latest_filename)
            
            if os.path.exists(src_file_latest):
                shutil.copy2(src_file_latest, os.path.join(target_season_dir, latest_filename))
                source_index.append({
                    'season': season,
                    'region': region,
                    'file_used': latest_filename,
                    'source_type': 'latest'
                })
                logger.info(f"Season {season} | Region {region} | Used latest")
            else:
                # 2. Fallback to the latest weekly file
                region_lower = region.lower()
                # Find all files for this region (case insensitive check for the prefix)
                weekly_files = [f for f in os.listdir(season_path) 
                                if f.lower().startswith(region_lower + "-") 
                                and f.endswith("-ILI.csv") 
                                and "latest" not in f]
                
                if weekly_files:
                    # Sort lexicographically (usually handles YYYY_WW correctly)
                    weekly_files.sort()
                    best_weekly = weekly_files[-1]
                    src_file_weekly = os.path.join(season_path, best_weekly)
                    shutil.copy2(src_file_weekly, os.path.join(target_season_dir, best_weekly))
                    source_index.append({
                        'season': season,
                        'region': region,
                        'file_used': best_weekly,
                        'source_type': 'weekly_fallback'
                    })
                    logger.info(f"Season {season} | Region {region} | Fallback to {best_weekly}")
                else:
                    logger.warning(f"No ILI data found for {region} in season {season}")

    # Save source index
    if source_index:
        df_index = pd.DataFrame(source_index)
        df_index.to_csv(SOURCE_INDEX_FILE, index=False)
        logger.info(f"Generated source index: {SOURCE_INDEX_FILE}")

if __name__ == "__main__":
    ingest_ili_data()
