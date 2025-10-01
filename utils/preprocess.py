from nltk import ngrams
from tqdm import tqdm
import itertools
from char_converter import CharConverter
from pypinyin import pinyin, lazy_pinyin, Style
import pandas as pd
import os

N_GRAM = 3
converter = CharConverter('v2t')

stroke_dict = {}
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
dict_path = os.path.join(script_dir, "dict_chinese_stroke.txt")
with open(dict_path, "r", encoding="utf-8") as file:
    for line in file:
        l = line.split(' ')
        c, v = l[0], l[1].strip()
        stroke_dict[c] = v

def initial_process_character(character, split_by_char=False):
    if isinstance(character, float):
        return []
    characters = list(character)

    if "（" in characters and "）" in characters:
        start_index = characters.index("（")
        end_index = characters.index("）")
        combined_char = "".join(characters[start_index+1:end_index])
        combined_char = combined_char.replace("+", "")
        
        characters = characters[:start_index] + [combined_char] + characters[end_index+1:]

    stroke_res = []
    pinyin_res = []
    for char in characters:
        char_res = []
        char_res.append("*")
        if len(char) > 1:
            for sub_char in char:
                sub_char_converted = converter.convert(sub_char)
                if sub_char == "？" or sub_char == "?" or sub_char == " ":
                    char_res.append("?")
                    pinyin_res.append("?")
                else:
                    char_res.extend(list(stroke_dict.get(sub_char, ["?"])))
                    char_res.extend(list(stroke_dict.get(sub_char_converted, ["?"])))
                    pinyin_res.extend(lazy_pinyin(sub_char))
            char_res.append("*")
        else:
            char_converted = converter.convert(char)
            if char == "？" or char == "?" or char == " ":
                char_res.append("?")
                pinyin_res.append("?")
            else:
                char_res.extend(list(stroke_dict.get(char, ["?"])))
                char_res.extend(list(stroke_dict.get(char_converted, ["?"])))
                pinyin_res.extend(lazy_pinyin(char))
            char_res.append("*")
            
        trigrams = list(ngrams(char_res, N_GRAM))
        trigrams_flat = [''.join(trigram) for trigram in trigrams]
        if not split_by_char:
            stroke_res.extend(trigrams_flat)
        else:
            stroke_res.append(trigrams_flat)
    return (stroke_res, pinyin_res)

def process_row(x, match_col_list):
    stroke_by_col = [initial_process_character(x[col], split_by_char=True)[0] for col in match_col_list]
    pinyin_by_col = [initial_process_character(x[col])[1] for col in match_col_list]
    match_col_stroke_list = [f'{col}_stroke' for col in match_col_list]
    match_col_pinyin_list = [f'{col}_pinyin' for col in match_col_list]
    val_dict = dict(zip(match_col_stroke_list + match_col_pinyin_list, stroke_by_col + pinyin_by_col))
    return val_dict

def process_df(raw_df, match_col_list):
    df = raw_df
    
    for col in match_col_list:
        df[col] = df[col].fillna("?")

    tqdm.pandas(desc='ngraming...')
    processed_rows = df.progress_apply(lambda x: process_row(x, match_col_list), axis=1)
    return pd.concat([df.reset_index(), pd.DataFrame.from_records(list(processed_rows))], axis=1)