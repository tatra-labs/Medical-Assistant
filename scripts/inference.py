import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
CHECKPOINT_DIR = os.path.join(root_dir, 'results', 'checkpoint-1000')
PREFIX         = "question: "
MAX_LENGTH     = 1024
NUM_BEAMS      = 5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

DO_SAMPLE            = True       # switch from beam search to sampling
TEMPERATURE          = 1.0        # try values between 0.7–1.2
TOP_P                = 0.9        # try 0.8–0.95
TOP_K                = 50         # try 10–50

# Repeat control
REPETITION_PENALTY   = 1.2        # try 1.1–1.5
NO_REPEAT_NGRAM_SIZE = 3          # prevents any 3‑gram from repeating

# If you prefer beam search instead of sampling, set:
# DO_SAMPLE = False
# NUM_BEAMS = 5
NUM_BEAMS            = 1          # only used if DO_SAMPLE=False

INPUTS = [
    "what are marine toxins?",
]
def load_model_and_tokenizer(checkpoint_dir, device):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    config    = AutoConfig.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model     = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir, config=config)
    model.to(device)
    model.eval()
    return model, tokenizer

def run_inference(model, tokenizer, inputs):
    # Prepend prefix
    texts = [PREFIX + t for t in inputs]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    with torch.no_grad():
        gen_kwargs = {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "max_length":     MAX_LENGTH,
            "pad_token_id":   tokenizer.pad_token_id,
            "eos_token_id":   tokenizer.eos_token_id,
            # repeat control:
            "repetition_penalty":   REPETITION_PENALTY,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        }

        if DO_SAMPLE:
            # sampling-based decoding
            gen_kwargs.update({
                "do_sample":   True,
                "temperature": TEMPERATURE,
                "top_p":       TOP_P,
                "top_k":       TOP_K,
            })
        else:
            # beam search
            gen_kwargs.update({
                "do_sample":   False,
                "num_beams":   NUM_BEAMS,
                "early_stopping": True,
            })

        out_ids = model.generate(**gen_kwargs)

    outputs = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    return [o.strip() for o in outputs]

def main():
    model, tokenizer = load_model_and_tokenizer(CHECKPOINT_DIR, DEVICE)
    outputs = run_inference(model, tokenizer, INPUTS)

    for inp, out in zip(INPUTS, outputs):
        print(f">>> INPUT:  {inp}")
        print(f"<<< OUTPUT: {out}\n")

if __name__ == "__main__":
    main()