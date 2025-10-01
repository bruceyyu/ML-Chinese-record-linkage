import pandas as pd
import numpy as np
import os
from importlib.machinery import SourceFileLoader
from tqdm import tqdm

# Load preprocess module
preprocess = SourceFileLoader('preprocess', './utils/preprocess.py').load_module()

# Configuration
match_col_list = ["xing", "ming", "zihao", "diqu", "jigou_1", "jigou_2", "guanzhi_1", "ren_xian", "ren_sheng", "chushen_1"]

output_dir = "./processed_dataset"
os.makedirs(output_dir, exist_ok=True)

def load_and_filter_data():
    """Load and filter the original data"""
    print("Loading original data...")
    original_df = pd.read_csv("./CGED-Q 1850-1864.csv")
    
    print("Filtering data...")
    df = original_df[original_df['xing'].notna()]
    pd.DataFrame(df['assigned_edition'].unique().tolist()).to_csv(f"{output_dir}/edition_list.csv", index=False, header=False)
    
    print(f"Filtered data shape: {df.shape}")
    return df

def get_edition_list(df):
    """Get sorted list of unique assigned_edition values"""
    edition_list = sorted(df['assigned_edition'].unique())
    print(f"Found {len(edition_list)} unique editions")
    return edition_list

def process_edition(df, edition, output_dir):
    """Process a single edition and save the result"""
    print(f"Processing edition: {edition}")
    
    # Filter data for this edition
    edition_df = df[df['assigned_edition'] == edition].copy()
    print(f"Edition {edition} has {len(edition_df)} records")
    
    if len(edition_df) == 0:
        print(f"Skipping edition {edition} - no records")
        return
    
    # Process the data
    try:
        df_processed = preprocess.process_df(edition_df, match_col_list)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the processed data
        output_file = os.path.join(output_dir, f"df_to_match_edition_{edition}.parquet")
        df_processed.to_parquet(output_file)
        print(f"Saved edition {edition} to {output_file}")
        
    except Exception as e:
        print(f"Error processing edition {edition}: {e}")

def main():
    """Main function to process all editions"""
    # Load and filter data
    df = load_and_filter_data()
    
    # Get list of editions
    edition_list = get_edition_list(df)
    
    # Output directory
    
    # Process each edition
    for edition in tqdm(edition_list, desc="Processing editions"):
        process_edition(df, edition, output_dir)
    
    print("Processing complete!")
    
    # Save edition list for reference
    edition_info = {
        'edition_list': edition_list,
        'total_editions': len(edition_list),
        'output_directory': output_dir
    }
    
    import pickle
    with open(os.path.join(output_dir, 'edition_info.pkl'), 'wb') as f:
        pickle.dump(edition_info, f)
    
    print(f"Edition info saved. Total editions processed: {len(edition_list)}")

if __name__ == '__main__':
    main() 