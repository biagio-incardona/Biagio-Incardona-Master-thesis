import os
import pandas as pd
import datetime
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "ili_gold.csv")

def iso_to_date(year, week):
    """
    Convert ISO year and week to the Sunday of that week.
    """
    return datetime.datetime.strptime(f'{year}-W{week:02d}-7', "%G-W%V-%u").date()

def preprocess_ili_data(time_index='epidemic', fill_zeros=False):
    """
    Standardize raw ILI CSVs into a long-format CSV.
    
    Args:
        time_index: 'calendar' (regular weekly index) or 'epidemic' (concatenated observations).
        fill_zeros: Only used in 'calendar' mode. If True, fills gaps with 0.0.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    all_data = []
    
    if not os.path.exists(RAW_DATA_DIR):
        logger.error(f"Raw data directory not found: {RAW_DATA_DIR}")
        return

    # Iterate through seasons
    seasons = sorted([s for s in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, s))])
    for season in seasons:
        season_path = os.path.join(RAW_DATA_DIR, season)
        
        # Iterate through files in season
        for filename in os.listdir(season_path):
            if not filename.endswith("-ILI.csv"):
                continue
                
            # Extract region from filename (handle both latest and weekly formats)
            # format: region-latest-ILI.csv OR region-YYYY_WW-ILI.csv
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
                df['calendar_ds'] = [iso_to_date(int(a), int(s)) for a, s in zip(df['anno'], df['settimana'])]
                df['y'] = df['incidenza']
                df['region'] = region
                df['season'] = season
                
                all_data.append(df[['region', 'calendar_ds', 'y', 'season']])
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    if not all_data:
        logger.warning("No data found to process.")
        return

    # Concatenate
    full_df = pd.concat(all_data, ignore_index=True)
    full_df['calendar_ds'] = pd.to_datetime(full_df['calendar_ds'])
    
    # Sort and remove duplicates (prefer latest season if overlap exists)
    full_df = full_df.sort_values(['region', 'calendar_ds', 'season'], ascending=[True, True, False])
    full_df = full_df.drop_duplicates(subset=['region', 'calendar_ds'])
    
    processed_regions = []
    for region in full_df['region'].unique():
        region_df = full_df[full_df['region'] == region].copy().sort_values('calendar_ds')
        
        if time_index == 'calendar':
            # Reindex to full weekly frequency
            min_date = region_df['calendar_ds'].min()
            max_date = region_df['calendar_ds'].max()
            full_range = pd.date_range(start=min_date, end=max_date, freq='W-SUN')
            
            region_df = region_df.set_index('calendar_ds').reindex(full_range)
            region_df['region'] = region
            if fill_zeros:
                region_df['y'] = region_df['y'].fillna(0.0)
            
            region_df = region_df.reset_index().rename(columns={'index': 'ds'})
            # Ensure calendar_ds is also present and matches ds
            region_df['calendar_ds'] = region_df['ds']
            
        elif time_index == 'epidemic':
            # Concatenate observations into a regular week-by-week index (ds)
            # starting from the first observation's Sunday
            region_df = region_df.reset_index(drop=True)
            start_date = region_df['calendar_ds'].iloc[0]
            # Generate a synthetic 'ds' index with weekly frequency
            region_df['ds'] = [start_date + datetime.timedelta(weeks=i) for i in range(len(region_df))]
            region_df['ds'] = pd.to_datetime(region_df['ds'])
            
        processed_regions.append(region_df)
        
    final_df = pd.concat(processed_regions, ignore_index=True)
    
    # Ensure columns are ordered: region, ds, y, calendar_ds
    cols = ['region', 'ds', 'y', 'calendar_ds']
    final_df = final_df[cols]
    
    # Save to CSV
    final_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved standardized data to {OUTPUT_FILE} (Mode: {time_index})")
    logger.info(f"Total rows: {len(final_df)}")
    logger.info(f"Regions: {final_df['region'].unique().tolist()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess ILI data')
    parser.add_argument('--time-index', type=str, default='epidemic', choices=['calendar', 'epidemic'],
                        help='Time indexing strategy (default: epidemic)')
    parser.add_argument('--fill-zeros', action='store_true', help='Fill gaps with 0.0 in calendar mode')
    args = parser.parse_args()
    
    preprocess_ili_data(time_index=args.time_index, fill_zeros=args.fill_zeros)
