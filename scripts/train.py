from transformers import AutoModelForSequenceClassification, AutoTokenizer,DataCollatorWithPadding,TrainingArguments, Trainer,set_seed,EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from utils import *
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
labels = load_labels(config['paths']['labels'],config['label']['task'])


peft_config = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=False,
    r=config['lora']['r'],
    lora_alpha=config['lora']['alpha'],
    lora_dropout=config['lora']['dropout'],
    target_modules = config['lora']['target'],
)

def preprocess(examples):
    return tokenizer(examples[config['text']], padding=True, truncation=True)
model = AutoModelForSequenceClassification.from_pretrained(config['model']['path'],num_labels=len(labels.keys()))
if config['tokenized']:
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['path'])
    data = load_tokenized(os.path.join(config['tokenizer']['data'],config['text']))
else:
    tokenizer = AutoTokenizer.from_pretrained(config['model']['path'])
    tokenizer.add_special_tokens({'additional_special_tokens':config['tokenizer']['tokens']})
    print("Special Tokens in Tokenizer:", tokenizer.additional_special_tokens)
    data = load_data(config['paths']['data'],config['label']['target'],config['eval_pct'])
    data = data.map(preprocess,batched=True)
    save_tokenized(data,config['tokenizer']['data'] + config['text'])
    


model.resize_token_embeddings(len(tokenizer))
model = get_peft_model(model,peft_config)
model.print_trainable_parameters()
model.to(device)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer,return_tensors='pt')
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,         # Stop after 3 evaluation rounds without improvement
    early_stopping_threshold=0.01      # Minimum improvement to reset patience
)
training_args = TrainingArguments(
    output_dir=os.path.join(config['model']['checkpoints'],config['model']['name']),
    learning_rate=float(config['model']['lr']),
    per_device_train_batch_size=config['model']['batch'], 
    per_device_eval_batch_size=config['model']['batch'],
    num_train_epochs=config['model']['epochs'],
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=config['model']['eval_steps'], 
    gradient_accumulation_steps=config['model']['GA'],
    weight_decay=float(config['model']['decay']),
    logging_dir='/content/logs',
    dataloader_num_workers=4,
    save_strategy="steps",
    save_steps=200000,
    warmup_steps=1000,
    metric_for_best_model="f1",
    greater_is_better=True,   
    load_best_model_at_end=True,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data['train'],
    eval_dataset=data['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

trainer.train()
trainer.save_model(os.path.join(config['model']['save'],config['model']['name']))
tokenizer.save_pretrained(os.path.join(config['tokenizer']['save'],config['model']['name']))
