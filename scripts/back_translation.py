import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm  # Import tqdm for the progress bar

# Function to load the translation model and tokenizer
def get_model_tokenizer(src_lang: str, tgt_lang: str):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Check if GPU is available and move model to GPU if it is
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model, tokenizer

# Function to translate a batch of sentences
def translate_batch(texts, model, tokenizer):
    # Prepare the batch
    batch = tokenizer.prepare_seq2seq_batch(src_texts=texts, return_tensors="pt", padding=True)

    # Move the input tensors to GPU if available
    if torch.cuda.is_available():
        batch = {key: value.to('cuda') for key, value in batch.items()}
    
    # Generate translations
    translated = model.generate(**batch)
    
    # Decode the translated texts
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# Back translation function with batch processing and tqdm progress bar
def back_translate_batch(texts, batch_size=8):
    augmented_texts = []
    
    # Initialize tqdm with total number of batches
    with tqdm(total=len(texts)//batch_size, desc="Processing Batches", unit="batch") as pbar:
        for i in range(0, len(texts), batch_size):
            # Process texts in smaller batches
            batch = texts[i:i+batch_size]
            
            # Spanish → English
            model_es_en, tokenizer_es_en = get_model_tokenizer("es", "en")
            english_texts = translate_batch(batch, model_es_en, tokenizer_es_en)
            
            # English → Spanish
            model_en_es, tokenizer_en_es = get_model_tokenizer("en", "es")
            spanish_texts = translate_batch(english_texts, model_en_es, tokenizer_en_es)
            
            augmented_texts.extend(spanish_texts)
            
            # Update the progress bar after each batch
            pbar.update(1)
    
    return augmented_texts

# Function to augment and concatenate 'celebrity' profession rows with back-translation
def augment_and_concat_celebrities(df, batch_size=8):
    # Filter out the 'celebrity' profession rows
    celebrity_df = df[df['profession'] == 'celebrity']
    
    # Perform back-translation only for 'celebrity' rows
    celebrity_texts = celebrity_df['tweet'].tolist()
    augmented_texts = back_translate_batch(celebrity_texts, batch_size)
    
    # Create a new DataFrame with the augmented 'celebrity' rows
    augmented_celebrities = celebrity_df.copy()
    augmented_celebrities['tweet'] = augmented_texts
    
    # Combine the original 'celebrity' rows with the augmented rows
    df_combined = pd.concat([df, augmented_celebrities], ignore_index=True)
    
    return df_combined

# Example DataFrame

# Convert to DataFrame
df = pd.read_csv('../data/iberlef_2/politicES_phase_2_train_public.csv')

# Augment and concatenate the 'celebrity' profession rows with back-translation
augmented_df = augment_and_concat_celebrities(df, batch_size=32)
augmented_df.to_csv('../data/augmented_back.csv')
# Show the augmented DataFrame
print(augmented_df)
