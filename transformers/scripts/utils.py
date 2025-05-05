from transformers import AutoModel
from datasets import Dataset,DatasetDict
import pickle as pkl
import pandas as pd
import numpy as np
import evaluate
import yaml
import json 
import pickle as pkl

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
        text = row['text']
        return f"{profession_token} {text}"

def load_labels(path,task):
    with open (path,'rb') as f:
        return pkl.load(f)[task]

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(path,label,eval_pct):
    data = pd.read_csv(path)
    data = data[['ID','profession','ideology_binary','ideology_multiclass','tweet','Mtweet']]
    data = data.rename(columns={label:'label'})
    dataset = Dataset.from_pandas(data)
    return dataset.train_test_split(test_size=eval_pct) 

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

def save_pkl(path,object):
    with open(path, "wb") as f:
        pkl.dump(object,f)

def save_json(path,object):
    with open(path, "w") as f:
        json.dump(object, f,indent=2)

def create_profile(path):
    label2ideology = {
        0: "left",
        1: "right",
        2: "moderate_left",
        3: "moderate_right"
    }
    label2profession = {
        0: "celebrity",
        1: "journalist",
        2: "politician"
    }
    data = pd.read_csv(path)
    data = data.drop(columns=['Unnamed: 0.3','Unnamed: 0.2','Unnamed: 0.1'],errors='ignore')
    data['profile_bin'] = data['profession'].map(label2profession) + '-' + data['ideology_binary'].map(label2ideology)
    data['profile_multi'] = data['profession'].map(label2profession) + '-' + data['ideology_multiclass'].map(label2ideology)
    unique_bin = data['profile_bin'].unique()
    profile2idbin = {v: i for i, v in enumerate(sorted(unique_bin))}
    idbin2profile = {v: k for k, v in profile2idbin.items()}
    unique_multi = data['profile_multi'].unique()
    data['profile_bin'] = data['profile_bin'].map(profile2idbin)
    profile2idmulti = {v: i for i, v in enumerate(sorted(unique_multi))}
    idmulti2profile = {v: k for k, v in profile2idmulti.items()}
    data['profile_multi'] = data['profile_multi'].map(profile2idmulti)
    data.to_csv(path,index=False)
    mappings = {
        'profile2idbin':profile2idbin,
        'idbin2profile':idbin2profile,
        'profile2idmulti':profile2idmulti,
        'idmulti2profile':idmulti2profile
    }
    print(mappings)
    save_json('mappings.json',mappings)
    save_pkl('mappings.pkl',mappings)

def cluster_tweets_by_80(
    path: str,
    output_path: str,
    tweet_col='text',
    id_col='id',
    words_per_sample=450,
    tweets_per_cluster=80
):
    df = pd.read_csv(path)
    tweets = df[[id_col, tweet_col]].values.tolist()

    clusters = []
    for cluster_start in range(0, len(tweets), tweets_per_cluster):
        cluster_chunk = tweets[cluster_start:cluster_start + tweets_per_cluster]
        current_text = []
        current_ids = []
        word_count = 0

        for tweet_id, tweet in cluster_chunk:
            words = tweet.split()
            if word_count + len(words) > words_per_sample:
                # Save the current subsample
                if current_text:
                    clusters.append({
                        'text': " ".join(current_text),
                        'tweet_ids': ",".join(map(str, current_ids))
                    })
                # Start new subsample
                current_text = [tweet]
                current_ids = [tweet_id]
                word_count = len(words)
            else:
                current_text.append(tweet)
                current_ids.append(tweet_id)
                word_count += len(words)

        # Save the last chunk in the cluster
        if current_text:
            clusters.append({
                'text': " ".join(current_text),
                'tweet_ids': ",".join(map(str, current_ids))
            })

    result_df = pd.DataFrame(clusters)
    print(result_df.columns)
    result_df.to_csv(output_path, index=False)
    print(f"Saved {len(result_df)} clustered samples to {output_path}")
    return result_df


if __name__ == '__main__':
    #view_model("dccuchile/bert-base-spanish-wwm-cased")
    #create_profile('../../data/joined_back_para_syn.csv')
    #create_word_limited_pseudo_users('../../data/joined_back_para_syn.csv','../../data/grouped_sep_bin.csv')
    #cluster_tweets_by_80('../../data/iberlef_2/politicES_phase_2_train_public.csv','clusters_80_sub.csv')
    pass