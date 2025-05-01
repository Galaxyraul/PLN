import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import numpy as np

# Load BETO tokenizer and model
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Function to split long text clusters into subsamples
def split_text(text, max_tokens=450):
    words = text.split()
    chunks = []
    current = []
    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

# Example data format (use your actual data)
data = [{
    "text": " ".join(["Tweet example."] * 80),
    "label_binary": 1,  # 0=left, 1=right
    "label_multiclass": 2  # 0=left, 1=moderate_left, 2=moderate_right, 3=right
} for _ in range(100)]

# Preprocess into subsamples
def prepare_subsamples(data, label_type="label_binary"):
    samples = []
    for item in data:
        subs = split_text(item["text"])
        for sub in subs:
            samples.append({"text": sub, "label": item[label_type]})
    return Dataset.from_list(samples)

# Add LoRA to the model
def get_lora_model(model_name, num_labels):
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    return get_peft_model(base_model, config)

# Tokenization

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# Training routine
def train_model(dataset, num_labels, output_dir):
    train_ds, val_ds = dataset.train_test_split(test_size=0.1).values()
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = get_lora_model(MODEL_NAME, num_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer

# Voting with softmax confidence

def vote_predictions(text, trainer_binary, trainer_multiclass):
    subs = split_text(text)
    inputs = tokenizer(subs, return_tensors="pt", padding=True, truncation=True)
    logits_bin = trainer_binary.model(**inputs).logits
    logits_multi = trainer_multiclass.model(**inputs).logits

    # Soft voting
    probs_bin = softmax(logits_bin.detach().numpy(), axis=1)
    probs_multi = softmax(logits_multi.detach().numpy(), axis=1)

    final_bin = np.argmax(np.sum(probs_bin, axis=0))
    final_multi = np.argmax(np.sum(probs_multi, axis=0))

    # IDC: ensure consistency between binary and multiclass
    if final_bin == 0 and final_multi > 1:
        final_multi = 0  # left
    elif final_bin == 1 and final_multi < 2:
        final_multi = 3  # right

    return final_bin, final_multi

# Example usage
dataset_bin = prepare_subsamples(data, label_type="label_binary")
dataset_multi = prepare_subsamples(data, label_type="label_multiclass")

# Train binary and multiclass models
trainer_bin = train_model(dataset_bin, 2, "./beto_ideology_binary")
trainer_multi = train_model(dataset_multi, 4, "./beto_ideology_multi")

# Predict
text_sample = data[0]["text"]
pred_bin, pred_multi = vote_predictions(text_sample, trainer_bin, trainer_multi)
print("Predicted Ideology (Binary):", pred_bin)
print("Predicted Ideology (Multiclass):", pred_multi)
