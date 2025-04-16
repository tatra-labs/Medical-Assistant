import os, joblib, logging, argparse 

import pandas as pd 
import numpy as np 

from google.cloud import storage 

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
    AutoModelForSeq2SeqLM
)
from datasets import Dataset, load_metric 

logging.basicConfig(level=logging.INFO) 
parser = argparse.ArgumentParser() 

parser.add_argument(
    '--data_gcs_path',
    help="Dataset file on GCS",
    type=str
)

args = parser.parse_args()
arguments = args.__dict__ 

root_dir = "gs://medquqa-data"
base_model_name = "razent/SciFive-large-Pubmed_PMC"

# Load SciFive
config = AutoConfig.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

sanitized_dataset_path = arguments['data_gcs_path']
sanitized_df = pd.read_csv(sanitized_dataset_path)
logging.info("reading gs data: {}".format(sanitized_dataset_path))

sanitized_df = sanitized_df[['prompt', 'response']]
sanitized_df = sanitized_df.rename(columns={'prompt': 'question', 'response': 'answer'})
logging.info("processing dataset")

results_path = os.path.join(root_dir, 'results')
logs_path = os.path.join(root_dir, 'logs')

train_val_df, test_df = train_test_split(sanitized_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

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
logging.info("preparing datasets")

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
    
training_args = TrainingArguments(
    output_dir=results_path,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    fp16=True,
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
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

logging.info("start training...")
trainer.train()
logging.info("training finished")

test_results = trainer.evaluate(test_dataset) 
logging.info("test results: {}".format(test_results))

local_path = 'model.joblib'
joblib.dump(trainer.model, local_path)
logging.info("model dumped to local path : {}".format(local_path))

model_directory = os.environ['AIP_MODEL_DIR']
storage_path = os.path.join(model_directory, 'model.joblib')
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(local_path)
logging.info("model exported to : {}".format(storage_path))