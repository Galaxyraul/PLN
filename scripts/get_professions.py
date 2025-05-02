from utils import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer,set_seed
from peft import get_peft_model
import argparse 
import torch
import os

if torch.cuda.is_available():
    device = 'cuda'
else:
    exit('No cuda exiting')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--config','-c',type=str,required=True,help='File with the params of the training')
args = parser.parse_args()


config = load_config(args.config)
set_seed(config['seed'])

labels = load_labels(config['labels'])['label2profession']
def tokenize_fn(text):
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

data = load_data(config['paths']['data'])

model = AutoModelForSequenceClassification.from_pretrained(config['models']['base'])
tokenizer = AutoTokenizer.from_pretrained(config['paths']['tokenizer'])
model.resize_token_embeddings(len(tokenizer))

model = get_peft_model(model,config['models']['base'])

model.to(device)

model.eval()

modified_tweets = []

for _, row in data.iterrows():
    text = row[config['text']]
    inputs = tokenize_fn(text)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    # Get label token (e.g., "<JOURNALIST>")
    label_token = labels[str(pred)]
    modified_text = f"{label_token} {text}"
    modified_tweets.append(modified_text)

# Add to DataFrame
data['predicted_tweet'] = modified_tweets

# Save to CSV
data.to_csv(config['paths']['output'], index=False)