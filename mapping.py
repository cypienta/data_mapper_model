import argparse
import csv
import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Constants
MODEL = 'markusbayer/CySecBERT'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REGEX_TABLE_FILE = 'helper_files/table_regex.json'
COLUMN_REGEX_FILE = 'helper_files/column_regex.json'
LLM_MAPPING_FILE = 'helper_files/LLM_mapping.csv'

def load_json_file(file_path: str) -> Dict:
    """Load and return JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return {}

def get_embeddings(target: str, tokenizer, model) -> np.ndarray:
    """Get the embeddings of a target string using a tokenizer and model."""
    target_input_ids = torch.tensor([tokenizer.encode(target, add_special_tokens=True)]).to(DEVICE)
    with torch.no_grad():
        target_outputs = model(target_input_ids)
    return target_outputs[0][:, 0, :].float().cpu().numpy()

def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate the cosine similarity between two embeddings."""
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def apply_regex_mapping(df: pd.DataFrame, column_class: Dict[str, List], threshold: float) -> Dict[str, List]:
    """Apply regex mapping to the dataframe columns."""
    regex_patterns = load_json_file(REGEX_TABLE_FILE)
    
    for pattern_name, pattern in tqdm(regex_patterns, desc="Applying regex patterns"):
        compiled_pattern = re.compile(pattern)
        for column in df.columns:
            match_ratio = df[column].dropna().astype(str).apply(lambda x: bool(compiled_pattern.match(x))).mean()
            if match_ratio > threshold:
                print(f"{pattern_name} {column} percentage of strings that match the pattern: {match_ratio * 100:.2f}%")
                column_class[column].append((pattern_name, "REGEX TABLE"))
    
    return column_class

def build_regex_pattern(target: Dict[str, List[str]]) -> str:
    """Build a regex pattern from the given target dictionary."""
    pattern = r'^(?P<prefix>'
    pattern += '|'.join(item.upper() for item in target['prefix'])
    pattern += ')?(?P<separator1>[^a-zA-Z0-9]*)(?P<body>'
    pattern += '|'.join(item.upper() for item in target['body'])
    pattern += ')(?P<separator2>[^a-zA-Z0-9]*)(?P<suffix>'
    pattern += '|'.join(item.upper() for item in target['suffix'])
    pattern += ')?$'
    return pattern

def apply_column_regex(df: pd.DataFrame, column_class: Dict[str, List]) -> Dict[str, List]:
    """Apply regex mapping to column names."""
    column_regex = load_json_file(COLUMN_REGEX_FILE)
    
    for key, value in column_regex.items():
        pattern = build_regex_pattern(value)
        compiled_pattern = re.compile(pattern)
        for column in df.columns:
            if compiled_pattern.fullmatch(column.upper()):
                print(f"{column} matched with {key}")
                column_class[column].append((key, "REGEX COLUMN"))
    
    return column_class

def apply_LLM_mapping(df: pd.DataFrame, column_class: Dict[str, List], threshold: float) -> Dict[str, List]:
    """Apply LLM mapping to column names."""
    all_LLM = pd.read_csv(LLM_MAPPING_FILE).values.tolist()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE)
    model.eval()
    
    target_embeddings = [get_embeddings(target[1], tokenizer, model) for target in tqdm(all_LLM, desc="Computing target embeddings")]
    
    for column in tqdm(df.columns, desc="Applying LLM mapping"):
        candidate_embedding = get_embeddings(column, tokenizer, model)
        for idx, target_embedding in enumerate(target_embeddings):
            similarity = calculate_cosine_similarity(target_embedding, candidate_embedding)
            if similarity > threshold:
                print(f"{all_LLM[idx][1]} {column} similarity: {similarity:.2f}%")
                if all_LLM[idx][0] not in column_class[column]:
                    column_class[column].append((all_LLM[idx][0], "LLM COLUMN"))
    
    return column_class

def write_output(df: pd.DataFrame, column_class: Dict[str, List], output_file: str):
    """Write the output to a CSV file."""
    try:
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Column', "Mapping"])
            for key, value in column_class.items():
                output = [key, value]
                output.extend(df[key].dropna().sample(n=min(5, df[key].dropna().shape[0])).tolist())
                writer.writerow(output)
    except IOError:
        print(f"Error writing to file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Read a JSON file and map the values.")
    parser.add_argument("filename", type=str, help="The JSON file to map")
    parser.add_argument("--regex_threshold", type=float, default=0.9, help="Threshold for regex matching (default: 0.9)")
    parser.add_argument("--LLM_threshold", type=float, default=0.98, help="Threshold for LLM matching (default: 0.98)")
    parser.add_argument("--output_file", type=str, default="mappings.csv", help="Output file name (default: mappings.csv)")
    args = parser.parse_args()

    try:
        df = pd.read_json(args.filename, convert_dates=False)
    except FileNotFoundError:
        print(f"File not found: {args.filename}")
        return
    except ValueError:
        print(f"Invalid JSON in file: {args.filename}")
        return

    column_class = {column: [] for column in df.columns}
    column_class = apply_regex_mapping(df, column_class, args.regex_threshold)
    column_class = apply_column_regex(df, column_class)
    column_class = apply_LLM_mapping(df, column_class, args.LLM_threshold)
    write_output(df, column_class, args.output_file)

if __name__ == "__main__":
    main()