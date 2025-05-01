import pandas as pd

def show_stats(data_path):
    data = pd.read_csv(data_path)
    print(data.shape)
    print(data.columns)
    print(100*data['profession'].value_counts()/data['profession'].shape)
    print(data['ideology_binary'].value_counts()/data['ideology_binary'].shape)
    print(data['ideology_multiclass'].value_counts()/data['ideology_multiclass'].shape)
    return data

def join_datasets(path_1,path_2):
    data1 = pd.read_csv(path_1)
    data2 = pd.read_csv(path_2)

    data = pd.concat([data1,data2[data2['ID'] > 179999]],ignore_index=True)
    data.to_csv('../data/joined_back_para.csv')


#show_stats('../data/augmented_syn.csv')
#show_stats('../data/augmented_back.csv')
#show_stats('../data/augmented_syn_back.csv')
#show_stats('../data/augmented_para.csv')
#show_stats('../data/joined_back_para.csv')\
#show_stats('../data/joined_back_para_syn.csv')
#join_datasets('../data/augmented_back.csv','../data/augmented_para.csv')

def add_profession_token(row):
        author_type = row['profession'].upper()  
        profession_token = f"<{author_type}>" 
        text = row['tweet']
        return f"{profession_token} {text}"
path = '../data/augmented_back.csv'
data = pd.read_csv(path)
data['Mtweet'] = data.apply(add_profession_token, axis=1)
data.to_csv(path)