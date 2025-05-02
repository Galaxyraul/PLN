from utils import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from peft import get_peft_model
import argparse 
import torch
import os
import torch.nn.functional as F
import pandas as pd

if torch.cuda.is_available():
    device = 'cuda'
else:
    exit('No cuda exiting')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--config', '-c', type=str, required=True, help='YAML config file')
args = parser.parse_args()

# Load config and seed
config = load_config(args.config)
set_seed(config['seed'])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['paths']['tokenizer'])

# Load separate base models for binary and multiclass tasks
model_bin_base = AutoModelForSequenceClassification.from_pretrained(config['models']['bin_base'])
model_multi_base = AutoModelForSequenceClassification.from_pretrained(config['models']['multi_base'])

# Apply LoRA to each base model
model_bin = get_peft_model(model_bin_base, config['models']['bin']).to(device).eval()
model_multi = get_peft_model(model_multi_base, config['models']['multi']).to(device).eval()

# Load data
data = load_data(config['paths']['data'])

# Weighting hyperparameter (adjustable)
alpha = 0.6  # weight for multiclass model; (1 - alpha) for binary model

# Group indices (ideology groups)
group_A_ids = [1, 2]
group_B_ids = [3, 4]

# Store predictions
final_preds = []

for _, row in data.iterrows():
    text = row[config["text"]]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        # Get logits from binary and multiclass models
        logits_bin = model_bin(**inputs).logits[0]      # Binary classification logits [2]
        logits_multi = model_multi(**inputs).logits[0]  # Multiclass logits [num_classes]

        # Apply softmax to convert logits to probabilities
        probs_bin = F.softmax(logits_bin, dim=-1)
        probs_multi = F.softmax(logits_multi, dim=-1)

        # Group multiclass probabilities into binary-like groups
        prob_A = probs_multi[group_A_ids].sum()  # Sum probabilities of group A (e.g., [1, 2])
        prob_B = probs_multi[group_B_ids].sum()  # Sum probabilities of group B (e.g., [3, 4])

        # Weighted sum between binary model and multiclass model
        final_prob_0 = (1 - alpha) * probs_bin[0] + alpha * prob_A
        final_prob_1 = (1 - alpha) * probs_bin[1] + alpha * prob_B

        # Predict final label (either 0 or 1)
        final_label = 1 if final_prob_1 > final_prob_0 else 0
        final_preds.append(final_label)

# Add final predictions to the DataFrame and save it
data["final_prediction"] = final_preds
data.to_csv(config["paths"]["output"], index=False)
