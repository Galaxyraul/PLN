from utils import *
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from rank_bm25 import BM25Okapi

data = load_data('../../data/joined_back_para_syn.csv')
test = load_test('../../data/task/PoliticES_test_kaggle.csv')
X,y,vectorizer = vectorize(data,'stemmed')
test_vectors = vectorizer.transform(test['stemmed'])
predictions = test[['id']].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=33
)

label2ideology = {
    0:'left',
    1:'right'
}

# Define a parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization parameter for linear models
}


clf = LinearSVC(class_weight='balanced', verbose=1)
# Fit the grid search
clf.fit(X_train, y_train)
# Evaluate the best model
y_pred = clf.predict(X_test)
predictions['label'] = clf.predict(test_vectors)
print(predictions.head(1))
predictions['label'] = predictions['label'].map(label2ideology) 
predictions.to_csv('SVM.csv', index=False)
print('Linear SVM')
print(classification_report(y_test, y_pred, target_names=["left","right"]))

clf = LogisticRegression(class_weight='balanced', max_iter=1000,verbose=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
predictions['label'] = clf.predict(test_vectors)
predictions['label'] = predictions['label'].map(label2ideology) 
predictions.to_csv('Logistic_regression.csv', index=False)
print('Logistic Regression')
print(classification_report(y_test, y_pred, target_names=["left","right"]))

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
predictions['label'] = clf.predict(test_vectors)
predictions['label'] = predictions['label'].map(label2ideology) 
predictions.to_csv('Multinomial.csv', index=False)
print('Multinomial')
print(classification_report(y_test, y_pred, target_names=["left","right"]))

clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=33,verbose=1,n_jobs=8)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
predictions['label'] = clf.predict(test_vectors)
predictions['label'] = predictions['label'].map(label2ideology) 
predictions.to_csv('RF.csv', index=False)
print('Random Forest')
print(classification_report(y_test, y_pred, target_names=["left","right"]))


