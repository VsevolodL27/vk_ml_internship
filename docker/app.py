import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score

train_df = pd.read_csv('./train_df.csv')
test_df = pd.read_csv('./test_df.csv')

train_df.drop_duplicates(inplace=True)
train_df.dropna(subset=['target'], inplace=True)
train_df.fillna(np.median, inplace=True)

test_df.drop_duplicates(inplace=True)
test_df.dropna(subset=['target'], inplace=True)
test_df.fillna(np.median, inplace=True)

features_2 = [
    'feature_70',
    'feature_38',
    'feature_51',
    'feature_7',
    'feature_10',
    'feature_66',
    'feature_45',
    'feature_70',
    'feature_51',
    'feature_38',
    'feature_7',
    'feature_10',
    'feature_43',
    'feature_28',
    'feature_27',
    'feature_54',
    'feature_49',
    'feature_69',
    'feature_65',
    'feature_30',
    'feature_32',
    'feature_41',
    'feature_15'
]

X_train = train_df[features_2]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = train_df['target'].values

X_test = test_df[features_2]
X_test = scaler.transform(X_test)
y_test = test_df['target']

clf = LogisticRegression(solver='liblinear', penalty='l2')
clf.fit(X_train, y_train)

predictions = clf.predict_proba(X_test)[:, 1]

print(f'NDCG-score: {ndcg_score([y_test.values], [predictions])}')