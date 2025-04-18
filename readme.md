# Data Mapper Model
**Map any event, alert, log format to a single target format automatically with regex and llms.**
___
<p align="center">
<a> <img src="https://badges.frapsoft.com/os/v3/open-source.svg?v=103"></a>
<a> <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat"></a>
<a> <img src="https://img.shields.io/pypi/l/mia.svg"></a>
<a href="https://https://github.com/cypienta/data_mapper_model/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/cypienta/data_mapper_model"></a>
<a href="https://github.com/cypienta/data_mapper_model/graphs/contributors" alt="Contributors"> <img src="https://img.shields.io/github/contributors/ezzeldinadel/attack_flow_detector" /></a>
<a href="https://github.com/cypienta/data_mapper_model/graphs/stars" alt="Stars"><img src="https://img.shields.io/github/stars/cypienta/data_mapper_model" /></a>
<a href="https://github.com/cypienta/data_mapper_model"><img alt="GitHub forks" src="https://img.shields.io/github/forks/cypienta/data_mapper_model"></a>

<br>
 <p align="center">
   <a> <img src="https://img.shields.io/badge/langchain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" /></a>
  <a>  <img src="https://img.shields.io/badge/ChatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white" style="display: block; margin-left: auto; margin-right: auto;" /></a> 
   <a><img src="https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black" /></a>
   <a><img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /></a>
</p>
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
