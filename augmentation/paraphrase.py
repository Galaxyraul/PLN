import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Load model and tokenizer
model_name = "mrm8488/bert2bert_shared-spanish-finetuned-paus-x-paraphrasing"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def paraphrase_batch(texts, num_return_sequences=5, num_beams=10):
    input_texts = [f"parafrasea: {t}" for t in texts]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Group outputs per input text
    grouped = [decoded[i * num_return_sequences:(i + 1) * num_return_sequences] for i in range(len(texts))]
    return grouped

def augment_paraphrases_batched(df, n_iters=1, batch_size=8):
    celeb_df = df[df['profession'] == 'celebrity'].copy()
    texts = celeb_df['tweet'].tolist()
    metadata = celeb_df.drop(columns=['tweet']).to_dict(orient='records')

    all_augmented_rows = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Paraphrasing in batches"):
        batch_texts = texts[i:i + batch_size]
        batch_meta = metadata[i:i + batch_size]

        try:
            paraphrased_groups = paraphrase_batch(batch_texts, num_return_sequences=n_iters)
            for meta, paraphrases in zip(batch_meta, paraphrased_groups):
                for p in paraphrases:
                    row = meta.copy()
                    row['tweet'] = p
                    all_augmented_rows.append(row)
        except Exception as e:
            print(f"Batch error: {e}")
            continue

    augmented_df = pd.DataFrame(all_augmented_rows)
    return pd.concat([df, augmented_df], ignore_index=True)

df = pd.read_csv('../data/iberlef_2/politicES_phase_2_train_public.csv')
augmented_df = augment_paraphrases_batched(df, batch_size=32)
augmented_df.to_csv('../data/augmented_para.csv')