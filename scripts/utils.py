from transformers import AutoModel
from datasets import Dataset,DatasetDict
import pickle as pkl
import pandas as pd
import numpy as np
import evaluate
import yaml

def show_dataframe(data_path):
    data = pd.read_csv(data_path)
    print(data.shape)
    print(data.columns)
    print(100*data['profession'].value_counts()/data['profession'].shape)
    print(data['ideology_binary'].value_counts()/data['ideology_binary'].shape)
    print(data['ideology_multiclass'].value_counts()/data['ideology_multiclass'].shape)
    return data

def join_show_dataframes(path_1,path_2):
    data1 = pd.read_csv(path_1)
    data2 = pd.read_csv(path_2)

    data = pd.concat([data1,data2[data2['ID'] > 179999]],ignore_index=True)
    data.to_csv('../data/joined_back_para.csv')

def add_profession_token(row):
        author_type = row['profession'].upper()  
        profession_token = f"<{author_type}>" 
        text = row['tweet']
        return f"{profession_token} {text}"

def load_labels(path,task):
    with open (path,'rb') as f:
        return pkl.load(f)[task]

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_train_data(path,label,eval_pct):
    data = pd.read_csv(path)
    data = data[['ID','profession','ideology_binary','ideology_multiclass','tweet','Mtweet']]
    data = data.rename(columns={label:'label'})
    dataset = Dataset.from_pandas(data)
    return dataset.train_test_split(test_size=eval_pct) 

def load_data(path):
    return pd.read_csv(path)

def load_tokenized(path):
    return DatasetDict.load_from_disk(path)

def save_tokenized(data,path):
    data.save_to_disk(path)

def view_model(path):
    model = AutoModel.from_pretrained(path)
    print(model)

accuracy = evaluate.load("accuracy")
f1_score = evaluate.load("f1")
def compute_metrics(eval_pred):
    # get predictions
    predictions, labels = eval_pred

    # predict most probable class
    predicted_classes = np.argmax(predictions, axis=1)
    # compute accuracy
    acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'],3)

    f1 = np.round(f1_score.compute(predictions=predicted_classes, references=labels, average='macro')['f1'],3)
    return {"Accuracy": acc,"F1" : f1}


if __name__ == '__main__':
    view_model("dccuchile/bert-base-spanish-wwm-cased")