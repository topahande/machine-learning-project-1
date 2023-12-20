import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# parameters

n_splits = 5
output_file = 'rf_model_diabetes.bin'


# data preparation

df = pd.read_csv('data/diabetes_risk_prediction_dataset.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')
numerical_columns = list(df.dtypes[df.dtypes == 'int64'].index)
df['class'] = (df['class'] == 'Positive').astype(int)
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

#print(categorical_columns)
#print(numerical_columns)

# training 

def train(df_train, y_train):
    dicts = df_train[categorical_columns + numerical_columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    max_depth = 15
    min_samples_leaf = 1
    n_estimators=250
    model = RandomForestClassifier(n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=1)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df[categorical_columns + numerical_columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# validation

print('Doing cross validation')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train['class'].values
    y_val = df_val['class'].values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1


print('validation results:')
print('AUC %.3f +- %.3f' % (np.mean(scores), np.std(scores)))


# training the final model

print('training the final model')

dv, model = train(df_full_train, df_full_train['class'].values)
y_pred = predict(df_test, dv, model)

y_test = df_test['class'].values
auc = roc_auc_score(y_test, y_pred)

print(f'AUC for test data = {auc}')


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')