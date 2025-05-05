from utils import *
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

ideology2label = {
    'left':0,
    'right':1
}
data = load_data('../data/clusters_80_sub.csv')
test = load_test('../data/clustered_test.csv')
#X,y,vectorizer = vectorize(data,'stemmed','ideology_binary',ideology2label) #ideology_binary,ideology_multiclass,profession
#test_vectors = vectorizer.transform(test['stemmed'])
X = vectorize_with_sbert(data)
y = data['ideology_binary'].map(ideology2label)
test_vectors = vectorize_with_sbert(test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=33
)

label2ideology = {
    0:'left',
    1:'right'
}

label2profession = {
    0: "celebrity",
    1: "journalist",
    2: "politician"
}

models = {
    'LinearSVM': LinearSVC(class_weight='balanced',C=1.0),
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'MultinomialNB': MultinomialNB(),
    'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=8, random_state=33)
}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    grouped_preds = clf.predict(test_vectors)
    grouped_df = test[['tweet_ids']].copy()
    grouped_df['label'] = grouped_preds
    grouped_df['label'] = grouped_df['label'].map(label2ideology)

    # Expand grouped predictions to tweet-level predictions
    expanded_rows = []
    for _, row in grouped_df.iterrows():
        label = row['label']
        id_group = row['tweet_ids']

        # Ensure id_group is a list (from string if needed)
        if isinstance(id_group, str):
            id_group = eval(id_group)
        if isinstance(id_group, int):
            id_group = [id_group] 
        for tweet_id in id_group:
            expanded_rows.append({'id': tweet_id, 'label': label})

    # Final tweet-level predictions
    final_df = pd.DataFrame(expanded_rows)
    final_df.to_csv(f'../predictions/{name}-clusters-encoder.csv', index=False)

    # Report on validation set
    print(f"\n{name}")
    print(classification_report(y_test, y_pred))

