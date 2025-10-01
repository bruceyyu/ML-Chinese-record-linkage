import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from collections import defaultdict
import time
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "word2vec_stroke_ngram_3_model_large.bin")
stroke_vec_model = Word2Vec.load(model_path)
avg_emb = np.array([stroke_vec_model.wv[key] for key in stroke_vec_model.wv.key_to_index.keys()]).mean(axis=0)

def get_stroke_ngram_emb(stroke):
    if stroke in stroke_vec_model.wv.key_to_index.keys():
        return stroke_vec_model.wv[stroke]
    else:
        return avg_emb

def get_stroke_list_emb(stroke_list):
    return np.vstack([get_stroke_ngram_emb(stroke) for stroke in stroke_list]).mean(axis=0)

def get_char_list_emb(char_list):
    return np.vstack([get_stroke_list_emb(char) for char in char_list]).mean(axis=0)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.neighbors import NearestNeighbors

def safe_flatten(char_sequence):
    if len(char_sequence) == 0:
        return []
    if isinstance(char_sequence[0], np.ndarray):
        return np.concatenate(char_sequence)
    return sum(char_sequence, [])  # Original list-based approach

def get_pinyin_jaccard_mat(left_df, right_df, block_col_pinyin_list):
    mlb = MultiLabelBinarizer()
    mlb.fit([safe_flatten(x) for x in pd.concat([left_df, right_df])[block_col_pinyin_list].values])
    left_pinyin_emb = mlb.transform([safe_flatten(x) for x in left_df[block_col_pinyin_list].values])
    right_pinyin_emb = mlb.transform([safe_flatten(x) for x in right_df[block_col_pinyin_list].values])
    
    j_sim_mat = pairwise_distances(left_pinyin_emb, right_pinyin_emb)
    return j_sim_mat
    
def get_stroke_cosine_mat(left_df, right_df, block_col_stroke_list):
    left_stroke_emb = np.array([
        get_char_list_emb(safe_flatten(char_list)) 
        for char_list in left_df[block_col_stroke_list].values
    ], dtype=np.float32)
    
    right_stroke_emb = np.array([
        get_char_list_emb(safe_flatten(char_list))
        for char_list in right_df[block_col_stroke_list].values
    ], dtype=np.float32)
    return cosine_similarity(left_stroke_emb, right_stroke_emb)

def get_pinyin_jaccard_topk(left_df, right_df, block_col_pinyin_list, k=100):
    """Returns (distances, indices) for top-k most similar items (smallest distance)"""    
    mlb = MultiLabelBinarizer()
    mlb.fit([safe_flatten(x) for x in pd.concat([left_df, right_df])[block_col_pinyin_list].values])
    left_pinyin_emb = mlb.transform([safe_flatten(x) for x in left_df[block_col_pinyin_list].values]).astype(bool)
    right_pinyin_emb = mlb.transform([safe_flatten(x) for x in right_df[block_col_pinyin_list].values]).astype(bool)
    nn = NearestNeighbors(n_neighbors=min(k, len(right_df)), metric='jaccard', n_jobs=1)
    nn.fit(right_pinyin_emb)
    
    distances, indices = nn.kneighbors(left_pinyin_emb)
    return distances, indices

def get_stroke_cosine_topk(left_df, right_df, block_col_stroke_list, k=100):
    """Returns (similarities, indices) for top-k most similar items"""

    left_stroke_emb = np.array([
        get_char_list_emb(safe_flatten(char_list)) 
        for char_list in left_df[block_col_stroke_list].values
    ], dtype=np.float32)
    
    right_stroke_emb = np.array([
        get_char_list_emb(safe_flatten(char_list))
        for char_list in right_df[block_col_stroke_list].values
    ], dtype=np.float32)
    
    nn = NearestNeighbors(n_neighbors=min(k, len(right_df)), metric='cosine', n_jobs=1)
    nn.fit(right_stroke_emb)
    
    distances, indices = nn.kneighbors(left_stroke_emb)
    similarities = 1 - distances  # Convert to similarities
    
    return similarities, indices

def get_blocking(left_df, right_df, block_col_list, top_k, guaranteed_match_col_list=None):
    block_col_stroke_list = [f'{col}_stroke' for col in block_col_list]
    block_col_pinyin_list = [f'{col}_pinyin' for col in block_col_list]
    
    block_res_list = []
    correct = 0
    wrong = 0

    k_buffer = top_k * 2  # Buffer to ensure we have enough candidates after filtering
    stroke_sims, stroke_indices = get_stroke_cosine_topk(left_df, right_df, block_col_stroke_list, k=k_buffer)
    pinyin_dists, pinyin_indices = get_pinyin_jaccard_topk(left_df, right_df, block_col_pinyin_list, k=k_buffer)

    guaranteed_pairs = defaultdict(list)
    if guaranteed_match_col_list:
        left_df.loc[:,'guarantee_key'] = left_df[guaranteed_match_col_list].astype(str).agg('_'.join, axis=1)
        right_df.loc[:,'guarantee_key'] = right_df[guaranteed_match_col_list].astype(str).agg('_'.join, axis=1)
        guarantee_map = right_df.groupby('guarantee_key').indices
        
        for _, row in left_df[['index', 'guarantee_key']].iterrows():
            key = row['guarantee_key']
            idx = row['index']
            if key in guarantee_map:
                guaranteed_pairs[idx] = guarantee_map[key]

        print("guaranteed_pairs: ", len(guaranteed_pairs.keys()))

    print("blocking start!")
    for i, left_row in tqdm(left_df.iterrows()):
        left_idx = left_row['index']

        guaranteed_indices_positions = guaranteed_pairs.get(left_idx, [])
        guaranteed_pairs[left_idx] = right_df.iloc[guaranteed_indices_positions]['index'].tolist()
        
        # Filter out guaranteed pairs from top-k results
        guaranteed_set = set(guaranteed_indices_positions)

        # Get top-k stroke indices (excluding guaranteed)
        stroke_top_positions = []
        for j, pos in enumerate(stroke_indices[i]):
            if pos not in guaranteed_set:
                stroke_top_positions.append(pos)
            if len(stroke_top_positions) >= top_k:
                break
        
        # Get top-k pinyin indices (excluding guaranteed)
        pinyin_top_positions = []
        for j, pos in enumerate(pinyin_indices[i]):
            if pos not in guaranteed_set:
                pinyin_top_positions.append(pos)
            if len(pinyin_top_positions) >= top_k:
                break
        
        # Combine results
        top_idx = list(set(stroke_top_positions) | set(pinyin_top_positions))
        
        
        block_res_list.append({"left_id": i, "blocked_right_id": top_idx})

        left_record = left_df.iloc[i]
        right_blocked_records = right_df.iloc[top_idx]

    return block_res_list, guaranteed_pairs

def get_ngram_similarity(left_ngram, right_ngram, index=0):
    """Helper function to calculate n-gram similarity for given index."""
    left_ngram_char = left_ngram[index] if index < len(left_ngram) else ['*?*']
    right_ngram_char = right_ngram[index] if index < len(right_ngram) else ['*?*']

    if '*?*' in [left_ngram_char[0], right_ngram_char[0]]:
        return -1

    return cosine_similarity(
        get_stroke_list_emb(left_ngram_char).reshape(1, -1), 
        get_stroke_list_emb(right_ngram_char).reshape(1, -1)
    )[0][0]
    
def get_model_feature(left_df, right_df, block_res_list, match_col_list):
    match_col_stroke_list = [f'{col}_stroke' for col in match_col_list]
    match_col_pinyin_list = [f'{col}_pinyin' for col in match_col_list]

    feature_list, original_feature_list = [], []
    feature_dict_keys =  ["1_" + key for key in match_col_list] + ["2_" + key for key in match_col_list]
    left_df_stroke_emb = left_df[match_col_stroke_list].apply(lambda x: [get_char_list_emb(col) for col in x.values]).values
    right_df_stroke_emb = right_df[match_col_stroke_list].apply(lambda x: [get_char_list_emb(col) for col in x.values]).values
    
    for block_pair in tqdm(block_res_list, desc="Generating features"):
        left_id = block_pair['left_id']
        blocked_right_id = block_pair['blocked_right_id']
        
        left_record = left_df.iloc[left_id]
        right_blocked_records = right_df.iloc[blocked_right_id]
        ### Collect for training
        left_stroke_match_emb = np.vstack(left_df_stroke_emb[left_id])
        
        for right_id, right_record in right_blocked_records.iterrows():
            idx_1 = int(left_record['index'])
            idx_2 = int(right_record['index'])
            if idx_1 == idx_2:
                continue
            right_stroke_match_emb = np.vstack(right_df_stroke_emb[right_id])

            # Calculate stroke feature similarities
            stroke_feature = np.diagonal(cosine_similarity(left_stroke_match_emb, right_stroke_match_emb)).copy()
            
            stroke_feature[(left_record[match_col_list].str.match(r'^(\?\s?)+$')) | 
                           (right_record[match_col_list].str.match(r'^(\?\s?)+$'))] = -1
            
            # Calculate pinyin feature similarities using Jaccard index
            pinyin_feature = np.array([
                jaccard_index(
                    left_record[match_col_pinyin_list].astype(str).tolist()[i], 
                    right_record[match_col_pinyin_list].astype(str).tolist()[i]
                ) 
                for i in range(len(match_col_pinyin_list))
            ])
            pinyin_feature[(left_record[match_col_list].str.match(r'^(\?\s?)+$')) | 
                           (right_record[match_col_list].str.match(r'^(\?\s?)+$'))] = -1
            
            # Build a feature dictionary
            feature_dict = dict(zip(feature_dict_keys, 
                                    left_record[match_col_list].tolist() + 
                                    right_record[match_col_list].tolist()))
            
            # Calculate the 'pinji' difference. The public dataset does not have this column. If you want to use this, you need to add the pinji_numeric column to the dataset.
            
            # left_pin, right_pin = left_record['pinji_numeric'], right_record['pinji_numeric']
            # left_pin = 10 if left_pin == 0 else left_pin
            # right_pin = 10 if right_pin == 0 else right_pin
            # pinji_diff = abs(right_pin - left_pin) if left_pin != -1 and right_pin != -1 else -1
            # pinji_lower = min(left_pin, right_pin)

            # Calculate the difference in name character counts ('ming')
            ming_cnt_diff = abs(len(right_record['ming']) - len(left_record['ming']))
            
            # Process and calculate similarities for first and second characters of names
            left_ming_ngram = left_record['ming_stroke']
            right_ming_ngram = right_record['ming_stroke']
            
            ming_1_sim = get_ngram_similarity(left_ming_ngram, right_ming_ngram, index=0)
            ming_2_sim = get_ngram_similarity(left_ming_ngram, right_ming_ngram, index=1) if len(left_ming_ngram) > 1 and len(right_ming_ngram) > 1 else -1
            
            same_year = int(left_record['year'] == right_record['year'])
            # Combine all features
            # extra_feat = [pinji_diff, pinji_lower, ming_cnt_diff, ming_1_sim, ming_2_sim, same_year]
            extra_feat = [ming_cnt_diff, ming_1_sim, ming_2_sim, same_year]
            feature = np.concatenate((stroke_feature, pinyin_feature, extra_feat), axis=0)
            feature_list.append(feature)
            
            # Add additional features to the dictionary
            feature_dict.update({
                # '1_pinji': float(left_record['pinji_numeric']),
                # '2_pinji': float(right_record['pinji_numeric']),
                '1_idx': idx_1,
                '2_idx': idx_2,
                '1_year': float(left_record['year']),
                '2_year': float(right_record['year'])
            })
            
            original_feature_list.append(feature_dict)
            
    return feature_list, original_feature_list

def jaccard_index(x, y):
    x_set, y_set = set(x), set(y)
    intersection = len(x_set & y_set)
    union = len(x_set | y_set)
    return intersection / union