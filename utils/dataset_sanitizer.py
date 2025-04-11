import os 

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
import pandas as pd

global total_tokens
total_tokens = 0

root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
print(root_dir)

dataset_path = os.path.join(root_dir, 'data', 'clean_data.csv')
print(dataset_path)

sanitized_dataset_path = os.path.join(root_dir, 'data', 'sanitized_data.csv')

base_model_path = os.path.join(root_dir, 'models', 'SciFive-large-Pubmed_PMC')

# Load SciFive
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)

config = AutoConfig.from_pretrained(base_model_path)
max_length = config.n_positions 
print(max_length)


df = pd.read_csv(dataset_path)

# Define function to check token lengths
def check_token_lengths(row):
    global total_tokens
    question = "question: "+ row['prompt']
    answer = row['response']
    question_tokens = tokenizer(question, truncation=False)
    answer_tokens = tokenizer(answer, truncation=False)
    question_length = len(question_tokens['input_ids'])
    answer_length = len(answer_tokens['input_ids'])
    total_tokens = total_tokens + question_length + answer_length
    return question_length <= max_length and answer_length <= max_length

sanitized_df = df[df.apply(check_token_lengths, axis=1)] 
sanitized_df = sanitized_df.drop_duplicates(subset=['prompt', 'response']) 

print(total_tokens)

sanitized_df.to_csv(sanitized_dataset_path, index=False)
