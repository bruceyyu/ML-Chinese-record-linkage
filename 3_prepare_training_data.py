import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import json
import os
import multiprocessing as mp
from importlib.machinery import SourceFileLoader
block_match = SourceFileLoader('block_match', './utils/block_match.py').load_module()

import gc
import traceback

import concurrent.futures
from functools import partial
import time

match_col_list = ["xing", "ming", "zihao", "diqu", "jigou_1", "jigou_2", "guanzhi_1", "ren_xian", "ren_sheng", "chushen_1"]
guaranteed_match_col_list = ["xing", "ming", "ren_xian", "ren_sheng"]
block_col_list = ["xing", "ming"]
match_col_stroke_list = [f'{col}_stroke' for col in match_col_list]
match_col_pinyin_list = [f'{col}_pinyin' for col in match_col_list]
# The public dataset does not have this column. If you want to use the pinji_diff, pinji_lower, you need to add the pinji_numeric column to the dataset.
# model_feature_names = match_col_stroke_list + match_col_pinyin_list + ['pinji_diff', 'pinji_lower', 'ming_cnt_diff', 'ming_sim1', 'ming_sim2', 'same_year']
model_feature_names = match_col_stroke_list + match_col_pinyin_list + ['ming_cnt_diff', 'ming_sim1', 'ming_sim2', 'same_year']

NUM_WORKERS = mp.cpu_count() - 2
CHUNK_SIZE = 5

TOP_K = 10
WINDOW_SIZE = 3

TRAIN_EDITION_SAMPLE_SIZE = 10

EDITIONS_DIR = "./processed_dataset"
OUTPUT_DIR = "./processed_data_for_training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=====================starting=====================')

def load_editions_for_window(edition_list, start_idx, window_size):
    """Load only the editions needed for a specific window"""
    window_editions = edition_list[start_idx:start_idx + window_size]
    
    dfs = []
    for edition in window_editions:
        file_path = os.path.join(EDITIONS_DIR, f"df_to_match_edition_{edition}.parquet")
        if os.path.exists(file_path):
            print(f"Loading edition {edition}")
            df = pd.read_parquet(file_path)
            dfs.append(df)
        else:
            print(f"Warning: Missing edition file {file_path}")
    
    if not dfs:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df
    
# Separate function to handle each iteration
def process_edition(edition_idx, edition_list, block_col_list, match_col_list):
    print(f"Processing edition {edition_list[edition_idx]}")
    
    # Load left edition
    left_file = os.path.join(EDITIONS_DIR, f"df_to_match_edition_{edition_list[edition_idx]}.parquet")
    if not os.path.exists(left_file):
        print(f"Warning: Missing left edition file {left_file}")
        return None
    
    left_df = pd.read_parquet(left_file).reset_index()
    
    # Load right editions window
    right_edition_window = edition_list[edition_idx:edition_idx+WINDOW_SIZE]
    right_df = load_editions_for_window(edition_list, edition_idx, WINDOW_SIZE)
    
    if right_df is None:
        print(f"Warning: No data found for right window at edition {edition_list[edition_idx]}")
        return None

    right_df = right_df.reset_index()
    
    edition_idx_map = dict(zip(right_edition_window, range(1, len(right_edition_window)+1)))
    
    left_df['edition_idx'] = 0
    right_df['edition_idx'] = right_df['assigned_edition'].map(edition_idx_map)
    
    try:
        block_res_train_list, _ = block_match.get_blocking(left_df, right_df, block_col_list, TOP_K, guaranteed_match_col_list)
        
        # Get model features
        train_feature_list, train_original_feature_list = block_match.get_model_feature(left_df, right_df, block_res_train_list, match_col_list)
        
        # Clean up
        del left_df, right_df, block_res_train_list
        gc.collect()
        
        return ([i.tolist() for i in train_feature_list], train_original_feature_list)
        
    except Exception as e:
        print(f"Error processing edition {edition_list[edition_idx]}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def chunked_parallel_process(edition_list, block_col_list, match_col_list):
    edition_log_list = []
    total_tasks = len(edition_list)
    
    # Create chunks properly
    sampled_edition_idx_list = list(range(0, len(edition_list), len(edition_list) // TRAIN_EDITION_SAMPLE_SIZE))
    chunk_starts_idx_list = list(range(0, len(sampled_edition_idx_list), CHUNK_SIZE))

    chunks = [
        sampled_edition_idx_list[start:min(start + CHUNK_SIZE, len(sampled_edition_idx_list))] 
        for start in chunk_starts_idx_list
    ]

    print(chunks)
    # Process each chunk and save results immediately
    for chunk_idx, chunk_list in enumerate(tqdm(chunks, desc="Processing chunks")):
        print(f"Processing chunk {chunk_idx}: editions {chunk_list}")
        # Create a new process pool for each chunk
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit tasks for each edition in the chunk
            futures = []
            for idx in chunk_list:
                futures.append(
                    executor.submit(
                        process_edition,
                        idx,
                        edition_list,
                        block_col_list,
                        match_col_list
                    )
                )
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        edition_log_list.append(result)
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
            
            # Clean up resources while executor is still active
            futures.clear()
            gc.collect()
        
        # Force a short pause to ensure resources are freed
        time.sleep(1)
    return edition_log_list

def main():
    print('Starting data loading...')
    edition_list = pd.read_csv(f"{EDITIONS_DIR}/edition_list.csv", header=None).squeeze().tolist()
    
    print(f'Processing with {NUM_WORKERS} workers...')
    start_time = time.time()
    
    edition_log_list_parallel = chunked_parallel_process(
        edition_list, block_col_list, match_col_list
    )

    train_feature_list = []
    train_original_feature_list = []

    for edition_idx, log_data in enumerate(tqdm(edition_log_list_parallel)):
        log_train_feature_list, log_train_original_feature_list = log_data
        train_feature_list.extend(log_train_feature_list)
        train_original_feature_list.extend(log_train_original_feature_list)
    
    X_train = pd.DataFrame(train_feature_list, columns=model_feature_names)
    X_train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    with open(f"{OUTPUT_DIR}/train_original.json", 'w') as f:
        json.dump(train_original_feature_list, f)
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()