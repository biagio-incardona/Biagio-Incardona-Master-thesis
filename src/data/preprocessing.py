import os
import pandas as pd
import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "ili_gold.csv")

def iso_to_date(year, week):
    """
    Convert ISO year and week to the Sunday of that week.
    ISO week 1 is the week with the first Thursday of the year.
    We want Sunday (day 7 of the ISO week).
    """
    # From ISO year, week, day 7 (Sunday)
    # Using %G for ISO year and %V for ISO week to handle year boundaries correctly
    return datetime.datetime.strptime(f'{year}-W{week:02d}-7', "%G-W%V-%u").date()

def preprocess_ili_data():
    """
    Standardize raw ILI CSVs into a long-format CSV.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    all_data = []
    
    if not os.path.exists(RAW_DATA_DIR):
        logger.error(f"Raw data directory not found: {RAW_DATA_DIR}")
        return

    # Iterate through seasons
    for season in os.listdir(RAW_DATA_DIR):
        season_path = os.path.join(RAW_DATA_DIR, season)
        if not os.path.isdir(season_path):
            continue
            
        # Iterate through files in season
        for filename in os.listdir(season_path):
            if not filename.endswith("-ILI.csv"):
                continue
                
            # Extract region from filename
            region = filename.split('-')[0].lower()
            
            file_path = os.path.join(season_path, filename)
            try:
                df = pd.read_csv(file_path)
                
                # Check required columns
                required = ['anno', 'settimana', 'incidenza']
                if not all(col in df.columns for col in required):
                    logger.warning(f"Missing columns in {file_path}. Skipping.")
                    continue
                
                # Map to dates
                df['ds'] = df.apply(lambda row: iso_to_date(int(row['anno']), int(row['settimana'])), axis=1)
                df['y'] = df['incidenza']
                df['region'] = region
                
                all_data.append(df[['region', 'ds', 'y']])
                logger.debug(f"Processed {filename} from season {season}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    if not all_data:
        logger.warning("No data found to process.")
        return

    # Concatenate
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Sort and remove duplicates
    full_df['ds'] = pd.to_datetime(full_df['ds'])
    full_df = full_df.sort_values(['region', 'ds']).drop_duplicates(subset=['region', 'ds'])
    
    # Continuous time series: for each region, reindex to weekly frequency
    processed_regions = []
    for region in full_df['region'].unique():
        region_df = full_df[full_df['region'] == region].set_index('ds')
        
        # Determine the full range of weeks
        min_date = region_df.index.min()
        max_date = region_df.index.max()
        full_range = pd.date_range(start=min_date, end=max_date, freq='W-SUN')
        
        # Reindex and fill missing weeks with 0.0 (as summer reporting is often 0 or not done)
        # Actually, interpolation might be better for small gaps, but for summers 0.0 is common in ILI.
        # Let's use 0.0 for now as it's the most common convention when surveillance is inactive.
        region_df = region_df.reindex(full_range)
        region_df['region'] = region
        region_df['y'] = region_df['y'].fillna(0.0)
        
        region_df = region_df.reset_index().rename(columns={'index': 'ds'})
        processed_regions.append(region_df)
        
    final_df = pd.concat(processed_regions, ignore_index=True)
    
    # Save to CSV
    final_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved standardized data to {OUTPUT_FILE}")
    logger.info(f"Total rows: {len(final_df)}")
    logger.info(f"Regions: {final_df['region'].unique().tolist()}")

if __name__ == "__main__":
    preprocess_ili_data()
