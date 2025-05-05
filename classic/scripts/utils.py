import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
import nltk
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,SnowballStemmer
from tqdm import tqdm
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

tqdm.pandas()

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("spanish")
stop = stopwords.words('spanish')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model.eval()

def get_sentence_embeddings(texts, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

def vectorize_with_sbert(data, column='text'):
    texts = data[column].astype(str).tolist()
    X = get_sentence_embeddings(texts)
    return X

def lemmatize_text(text):
    return ' '.join(lemmatizer.lemmatize(word) for word in text)

def stem_text(text):
    return ' '.join(stemmer.stem(word) for word in text)
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove @ mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(text):
    return [word for word in text if word not in stop]

def load_test(path,profession=False):
    data = pd.read_csv(path)
    data['clean'] = data['text'].astype(str).progress_apply(clean_text)
    # Tokenize, remove stopwords, and lemmatize/stem the text
    print('Tokenizing')
    data['tokenized'] = data['clean'].progress_apply(tokenize_text)
    print('Stopwords')
    data['no_stopwords'] = data['tokenized'].progress_apply(remove_stopwords)
    print('Lematizing')
    data['lemmatized'] = data['no_stopwords'].progress_apply(lemmatize_text)
    print('Stemming')
    data['stemmed'] = data['no_stopwords'].progress_apply(stem_text)
    if profession:
        data['combined'] = data['stemmed'] + ' profesión: ' + data['profession'].astype(str).str.lower()
    
    return data

def load_data(path,profession=False):
    """Load data, clean it, and load the label mapping."""
    # Load the data
    data = pd.read_csv(path)
    print('Preprocessing text')
    # Clean the text
    data['clean'] = data['text'].astype(str).progress_apply(clean_text)
    
    # Tokenize, remove stopwords, and lemmatize/stem the text
    print('Tokenizing')
    data['tokenized'] = data['clean'].progress_apply(tokenize_text)
    print('Stopwords')
    data['no_stopwords'] = data['tokenized'].progress_apply(remove_stopwords)
    print('Lematizing')
    data['lemmatized'] = data['no_stopwords'].progress_apply(lemmatize_text)
    print('Stemming')
    data['stemmed'] = data['no_stopwords'].progress_apply(stem_text)
    
    # Combine the cleaned text with the profession information
    if profession:
        data['combined'] = data['stemmed'] + ' profesión: ' + data['profession'].astype(str).str.lower()
    
    return data

def vectorize(data,target,label,labels2id):
    vectorizer = TfidfVectorizer(stop_words=stop, max_features=10000, ngram_range=(1, 4))
    X = vectorizer.fit_transform(data[target])
    y = data[label].map(labels2id)
    return X,y,vectorizer

def load_labels(path):
    with open (path, 'rb') as f:
        return pkl.load(f)['label2ideology']