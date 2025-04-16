import os 
import sys 

import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
import torch 
from torch.utils.data import Dataset, DataLoader 
from torch.utils.tensorboard import SummaryWriter 

from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq, 
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainerCallback
)
from datasets import Dataset, load_metric 

root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
print(root_dir)

original_dataset_path = os.path.join(root_dir, 'data', 'mle_screening_dataset.csv')
print(original_dataset_path)

base_model_path = 'razent/SciFive-large-Pubmed_PMC'

# print(torch.cuda.is_available())  # Should print: True
# print(torch.cuda.get_device_name(0))  # Prints the name of your GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SciFive
config = AutoConfig.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
model = model.to(device)

model.gradient_checkpointing_enable()

sanitized_dataset_path = os.path.join(root_dir, 'data', 'sanitized_data.csv')
sanitized_df = pd.read_csv(sanitized_dataset_path)
sanitized_df = sanitized_df[['prompt', 'response']]
sanitized_df = sanitized_df.rename(columns={'prompt': 'question', 'response': 'answer'})

results_path = os.path.join(root_dir, 'results')
logs_path = os.path.join(root_dir, 'logs')

train_val_df, test_df = train_test_split(sanitized_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.025, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

max_length = config.n_positions 
prefix = "question: "

def preprocess(example):
    questions = [prefix + q for q in example["question"]]
    inputs = tokenizer(
        questions,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    targets = tokenizer(
        example["answer"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    example['input_ids'] = inputs.input_ids
    example['attention_mask'] = inputs.attention_mask
    example['labels'] = targets.input_ids
    
    return example

train_dataset = train_dataset.map(
    preprocess,
    batched=True,
    remove_columns=['question', 'answer']
)

val_dataset = val_dataset.map(
    preprocess,
    batched=True,
    remove_columns=['question', 'answer']
)

test_dataset = test_dataset.map(
    preprocess,
    batched=True,
    remove_columns=['question', 'answer']
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

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
    
class ClearMemoryCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

training_args = TrainingArguments(
    output_dir=results_path,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    eval_accumulation_steps=50,
    bf16=True,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir=logs_path,
    logging_steps=10, 
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100, 
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=30,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[ClearMemoryCallback()],
)

trainer.train()
trainer.save_model(results_path)

def cpu_evaluate(trainer, dataset):
    model.to("cpu")
    torch.cuda.empty_cache()
    results = trainer.evaluate(dataset)
    model.to("cuda")
    return results

test_results = cpu_evaluate(trainer, test_dataset)
print(test_results)