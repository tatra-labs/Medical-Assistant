import os 
import sys 

import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from torch.utils.data import Dataset, DataLoader 
from torch.utils.tensorboard import SummaryWriter 
from datasets import load_metric 

root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
print(root_dir)

original_dataset_path = os.path.join(root_dir, 'data', 'mle_screening_dataset.csv')
print(original_dataset_path)

base_model_path = os.path.join(root_dir, 'models', 'SciFive-large-Pubmed_PMC')

# Load SciFive
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)

sanitized_dataset_path = os.path.join(root_dir, 'data', 'sanitized_data.csv')
sanitized_df = pd.read_csv(sanitized_dataset_path)

class MedQuQADataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024):
        self.df = df 
        self.tokenizer = tokenizer 
        self.max_length = max_length 
        
    def __getitem__(self, idx):
        question, answer = self.df.iloc[idx]['prompt'], self.df.iloc[idx]['response']
        inputs = self.tokenizer(
            f"question: {question}",
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }
    
    def __len__(self):
        return len(self.df)
    

results_path = os.path.join(root_dir, 'results')
logs_path = os.path.join(root_dir, 'logs')

train_val_df, test_df = train_test_split(sanitized_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

train_dataset = MedQuQADataset(train_df, tokenizer)
val_dataset = MedQuQADataset(val_df, tokenizer)
test_dataset = MedQuQADataset(test_df, tokenizer)

# Metrics 
f1_metric = load_metric("f1", trust_remote_code=True)
bleu_metric = load_metric("bleu", trust_remote_code=True)
rouge_metric = load_metric("rouge", trust_remote_code=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred 
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True) 
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id) 
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True) 
    
    # Normalize predictions and labels for EM (strip whitespace)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Exact Match (EM)
    em_scores = [1 if pred == ref else 0 for pred, ref in zip(decoded_preds, decoded_labels)]
    em_result = {"exact_match": np.mean(em_scores)}
    
    # BLEU 
    bleu_preds = [pred.split() for pred in decoded_preds]
    bleu_refs = [[ref.split()] for ref in decoded_labels]
    bleu_result = bleu_metric.compute(predictions=bleu_preds, references=bleu_refs)
    
    # F1 
    f1_preds = [set(pred.split()) for pred in decoded_preds]
    f1_refs = [set(ref.split()) for ref in decoded_labels]
    f1_scores = []
    for pred, ref in zip(f1_preds, f1_refs):
        true_positives = len(pred & ref)
        precision = true_positives / len(pred) if pred else 0
        recall = true_positives / len(ref) if ref else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        f1_scores.append(f1)
    f1_result = {"f1": np.mean(f1_scores)}
    
    # ROUGE 
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "exact_match": em_result["exact_match"],
        "bleu": bleu_result["bleu"],
        "f1": f1_result["f1"],
        "rouge1": rouge_result["rouge1"].mid.fmeasure,
        "rouge2": rouge_result["rouge2"].mid.fmeasure,
        "rougeL": rouge_result["rougeL"].mid.fmeasure
    }
    
training_args = TrainingArguments(
    output_dir=results_path,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir=logs_path,
    logging_steps=10,  # Log every 10 steps
    evaluation_strategy="steps",
    eval_steps=10,  # Evaluate every 10 steps
    save_strategy="steps",
    save_steps=10,  # Checkpoint every 10 steps
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

test_results = trainer.evaluate(test_dataset) 

print(test_results)