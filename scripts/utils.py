import pandas as pd

def show_stats(data_path):
    data = pd.read_csv(data_path)
    print(data.shape)
    print(data.columns)
    print(100*data['profession'].value_counts()/data['profession'].shape)
    print(data['ideology_binary'].value_counts()/data['ideology_binary'].shape)
    print(data['ideology_multiclass'].value_counts()/data['ideology_multiclass'].shape)
    return data

def add_profession_token(row,authors_col,text_col):
        author_type = row[authors_col].upper()  
        profession_token = f"[{author_type}]" 
        text = row[text_col]
        return f"{profession_token} {text}"

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
show_stats('../data/joined_back_para_syn.csv')
#join_datasets('../data/augmented_back.csv','../data/augmented_para.csv')