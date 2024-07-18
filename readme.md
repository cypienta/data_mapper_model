# Data Mapper Model

## Description

This script reads a JSON file specified from the command line, and provides mappings for the json attributes

## Requirements

To run this tool, you need Python 3.6+ and the following libraries:

- pandas
- torch
- transformers
- scikit-learn
- tqdm

You can install the required libraries using the following command:


pip install -r requirements.txt


## Usage

To use the tool, run the following command:

python mapping.py <filename> [options]

### Arguments

- `filename`: The JSON file to map (required)

### Options

- `--regex_threshold`: Threshold for regex matching (default: 0.9)
- `--LLM_threshold`: Threshold for LLM matching (default: 0.98)
- `--output_file`: Output file name (default: mappings.csv)

## Example

python mapping.py data.json --regex_threshold 0.85 --LLM_threshold 0.95 --output_file results.csv

## Output

The tool generates a CSV file with the following columns:

- Column: The name of the column from the input JSON
- Mapping: The mapped categories and their sources (REGEX TABLE, REGEX COLUMN, or LLM COLUMN)
- Sample Data: Up to 5 sample values from the column

## Configuration Files

- `table_regex.json`: Contains regex patterns for table data matching
- `column_regex.json`: Contains regex patterns for column name matching
- `LLM_mapping.csv`: Contains target mappings for LLM-based similarity matching

## Note

This tool requires a CUDA-capable GPU for optimal performance when using LLM-based matching. If a GPU is not available, it will use the CPU, which may be significantly slower.
