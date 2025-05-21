import pandas as pd
import os

DATA_DIR = 'data'
BULK_DATA_PATH = os.path.join(DATA_DIR, 'karkidi_jobs_selenium_BULK_datascience.csv') # Verify 

#new_master_seen_urls.csv
MASTER_SEEN_URLS_PATH = os.path.join(DATA_DIR, 'master_seen_urls.csv')

df_bulk = pd.read_csv(BULK_DATA_PATH)
if 'Job_URL' in df_bulk.columns:
        unique_urls = df_bulk[
            (df_bulk['Job_URL'] != "Not found") & (df_bulk['Job_URL'].notna())
        ]['Job_URL'].unique()
        
        # Create a DataFrame for saving
        df_seen_urls = pd.DataFrame(unique_urls, columns=['Job_URL'])
        
        df_seen_urls.to_csv(MASTER_SEEN_URLS_PATH, index=False)
        
        print(f"Successfully created and populated '{MASTER_SEEN_URLS_PATH}' with {len(unique_urls)} unique URLs.")
else:
        print("Error: 'Job_URL' column not found .")
    
