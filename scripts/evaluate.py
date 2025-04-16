import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import Dataset
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

# Set up paths
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sanitized_dataset_path = os.path.join(root_dir, 'data', 'sanitized_data.csv')
results_path = os.path.join(root_dir, 'results')
logs_path = os.path.join(root_dir, 'logs')
base_model_path = 'razent/SciFive-large-Pubmed_PMC'

# Load configuration and tokenizer
config = AutoConfig.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
max_length = config.n_positions

# Load and preprocess test dataset
sanitized_df = pd.read_csv(sanitized_dataset_path)
sanitized_df = sanitized_df[['prompt', 'response']]
sanitized_df = sanitized_df.rename(columns={'prompt': 'question', 'response': 'answer'})
train_val_df, test_df = train_test_split(sanitized_df, test_size=0.002, random_state=42)
test_dataset = Dataset.from_pandas(test_df)

prefix = "question: "

def preprocess(example):
    questions = [prefix + q for q in example["question"]]
    inputs = tokenizer(
        questions,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    targets = tokenizer(
        example["answer"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    example['input_ids'] = inputs['input_ids']
    example['attention_mask'] = inputs['attention_mask']
    example['labels'] = targets['input_ids']
    return example

test_dataset = test_dataset.map(
    preprocess,
    batched=True,
    remove_columns=['question', 'answer']
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Handle predictions
    if isinstance(predictions, tuple):
        # Assume the first element contains the relevant data (e.g., logits or token IDs)
        predictions = predictions[0]
    if isinstance(predictions, (np.ndarray, torch.Tensor)):
        # Convert to list
        predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions.cpu().numpy().tolist()
    
    # Handle labels
    if isinstance(labels, tuple):
        labels = labels[0]
    if isinstance(labels, (np.ndarray, torch.Tensor)):
        labels = labels.tolist() if isinstance(labels, np.ndarray) else labels.cpu().numpy().tolist()
    
    # Flatten predictions if over-nested
    def flatten_predictions(preds):
        flattened = []
        for item in preds:
            # If item is a list of lists, take the first sequence or flatten further
            if isinstance(item, list) and all(isinstance(i, list) for i in item):
                # Handle case like [[[id1, id2, ...], ...], ...]
                for subitem in item:
                    if isinstance(subitem, list) and all(isinstance(i, (int, float)) for i in subitem):
                        flattened.append(subitem)
                    else:
                        raise ValueError(f"Unexpected nested structure in predictions: {subitem}")
            elif isinstance(item, list) and all(isinstance(i, (int, float)) for i in item):
                flattened.append(item)
            else:
                raise ValueError(f"Unexpected structure in predictions: {item}")
        return flattened
    
    predictions = flatten_predictions(predictions)
    
    # Validate predictions format
    if not all(isinstance(seq, list) and all(isinstance(id, (int, float)) for id in seq) for seq in predictions):
        raise ValueError("Predictions must be a list of token ID sequences")
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Process labels
    labels = [[tokenizer.pad_token_id if token == -100 else token for token in seq] for seq in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Normalize predictions and labels
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Exact Match
    em_scores = [1 if pred == ref else 0 for pred, ref in zip(decoded_preds, decoded_labels)]
    em_result = {"exact_match": np.mean(em_scores)}
    
    # BLEU
    bleu_refs = [[ref.split()] for ref in decoded_labels]  # list of list of list of tokens
    bleu_preds = [pred.split() for pred in decoded_preds]  # list of list of tokens
    bleu_score = corpus_bleu(bleu_refs, bleu_preds)
    bleu_result = {"bleu": bleu_score}
    
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
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for pred, ref in zip(decoded_preds, decoded_labels):
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)
    rouge_result = {key: np.mean(values) for key, values in rouge_scores.items()}
    
    return {
        "exact_match": em_result["exact_match"],
        "bleu": bleu_result["bleu"],
        "f1": f1_result["f1"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"]
    }

# Get and sort checkpoint directories
checkpoint_dirs = [d for d in os.listdir(results_path) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(results_path, d))]
checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))

# Training arguments for evaluation
training_args = TrainingArguments(
    output_dir=results_path,
    per_device_eval_batch_size=1,
    logging_dir=logs_path,
    report_to=["tensorboard"],
)

# Evaluate each checkpoint
for checkpoint_dir in checkpoint_dirs:
    step = int(checkpoint_dir.split("-")[1])
    model_path = os.path.join(results_path, checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Set the global step for TensorBoard logging
    trainer.state.global_step = step
    metrics = trainer.evaluate()
    print(f"Checkpoint {step}: {metrics}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()