import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from utils import load_config,load_labels
import argparse 
import os
from tqdm import tqdm
if torch.cuda.is_available():
    device = 'cuda'
else:
    exit('No cuda exiting')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--config','-c',type=str,required=True,help='File with the params of the training')
args = parser.parse_args()

config = load_config(args.config)
labels = load_labels(config['labels'],config['task'])
beto = AutoModelForSequenceClassification.from_pretrained(config['model']['beto']).cuda().eval()
tokenizer_beto = AutoTokenizer.from_pretrained(config['tokenizer']['beto'])
maria = AutoModelForSequenceClassification.from_pretrained(config['model']['maria']).cuda().eval()
tokenizer_maria =  AutoTokenizer.from_pretrained(config['tokenizer']['maria'])

# Load text data
df = pd.read_csv(config['data'])
texts = df[config["text"]].tolist()
if not config['clustered']:
    predictions = []
    predictions_df= df[['id']].copy()
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_class = torch.argmax(logits, dim=1).item()
        
        predictions.append(labels[pred_class])

    predictions_df["label"] = predictions
else:
    predictions = []
    id_list_flat = []

    for text, id_group in tqdm(zip(texts, df[config['ids']]), total=len(texts), desc="Predicting"):
        inputs_beto = tokenizer_beto(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        inputs_maria = tokenizer_maria(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            logits_beto = beto(**inputs_beto).logits
            logits_maria = maria(**inputs_maria).logits
        pred_class_beto = torch.argmax(logits_beto, dim=1).item()
        pred_class_maria = torch.argmax(logits_maria, dim=1).item()
        label_beto = labels[pred_class_beto]
        label_maria = labels[pred_class_maria]

        if label_beto == label_maria:
            label = label_beto
        else:
            logits = logits_beto + logits_maria
            pred = torch.argmax(logits,dim=1).item()
            label = labels[pred]
        # Parse the stringified list of IDs (e.g., "[1, 2, 3]") into a list of ints
        if isinstance(id_group, str):
            id_group = eval(id_group)  # Assumes format like "[1, 2, 3]"
        if isinstance(id_group, int):
            id_group = [id_group] 
        for id_ in id_group:
            id_list_flat.append(id_)
            predictions.append(label)
    predictions_df = pd.DataFrame({"id": id_list_flat, "label": predictions})

predictions_df.to_csv(config['output']+config['name']+'.csv', index=False)
print(f"Saved predictions to {config['output']}")
