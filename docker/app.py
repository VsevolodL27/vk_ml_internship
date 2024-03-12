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

counter = {
            'feature_70': 4, 'feature_38': 3, 'feature_10': 3, 'feature_43': 3, 'feature_49': 2, 'feature_69': 2,
            'feature_65': 2, 'feature_30': 2, 'feature_32': 2, 'feature_60': 2, 'feature_46': 2, 'feature_51': 1,
            'feature_7': 1, 'feature_66': 1, 'feature_45': 1, 'feature_28': 1, 'feature_27': 1, 'feature_54': 1,
            'feature_41': 1, 'feature_15': 1, 'feature_6': 1, 'feature_9': 1, 'feature_52': 1, 'feature_78': 1,
            'feature_25': 1, 'feature_21': 1, 'feature_24': 1, 'feature_47': 1, 'feature_29': 1, 'feature_77': 1
}

features_2 = []

for key, val in counter.items():
    features_2.extend([key] * val)

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

print(f'NDCG-score: {round(ndcg_score([y_test.values], [predictions]), 3)}')