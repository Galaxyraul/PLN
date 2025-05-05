import pandas as pd
import random
import spacy
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import nltk
from collections import defaultdict
# Download required resources if not already done
#nltk.download('omw-1.4')
#nltk.download('wordnet')

# Load Spanish spaCy model
nlp = spacy.load("es_core_news_sm")

def get_spanish_synonyms(word):
    synsets = wn.synsets(word, lang='spa')
    synonyms = set()
    for syn in synsets:
        for lemma in syn.lemma_names('spa'):
            if lemma.lower() != word.lower():
                synonyms.add(lemma.replace("_", " "))
    return list(synonyms)

def synonym_replacement(text, replace_prob=0.2):
    doc = nlp(text)
    new_words = []

    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and random.random() < replace_prob:
            synonyms = get_spanish_synonyms(token.text)
            if synonyms:
                new_word = random.choice(synonyms)
                new_words.append(new_word)
                continue
        new_words.append(token.text)

    return " ".join(new_words)

def augment_and_concat_celebrities_synonyms_multi_unique(df, replace_prob=1, n_iters=1):
    celebrity_df = df[df['profession'] == 'celebrity'].copy()
    augmented_rows = []

    # Track which sentences have already been generated per original
    seen_augmented = defaultdict(set)

    for _, row in tqdm(celebrity_df.iterrows(), total=len(celebrity_df), desc="Augmenting celebrities"):
        original_text = row['tweet']
        seen_texts = seen_augmented[original_text]

        attempts = 0
        generated = 0
        while generated < n_iters and attempts < n_iters * 3:
            new_text = synonym_replacement(original_text, replace_prob)
            if new_text != original_text and new_text not in seen_texts:
                seen_texts.add(new_text)
                new_row = row.copy()
                new_row['tweet'] = new_text
                augmented_rows.append(new_row)
                generated += 1
            attempts += 1  # Prevent infinite loop

    augmented_df = pd.DataFrame(augmented_rows)
    return pd.concat([df, augmented_df], ignore_index=True)

df = pd.read_csv('../data/joined_back_para.csv')

# Augment and concatenate the 'celebrity' profession rows with back-translation
augmented_df = augment_and_concat_celebrities_synonyms_multi_unique(df)
augmented_df.to_csv('../data/joined_back_para_syn.csv')