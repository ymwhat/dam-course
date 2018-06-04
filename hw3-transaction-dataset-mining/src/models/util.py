import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import train_test_split
import params
import pickle
import os
def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load((f))

def save_to_file(output, test_pred, filename):
    output['target'] = pd.Series(test_pred).apply(lambda x: 'Yes' if x else 'No')
    output.to_csv(params.output+filename, index=False, header=False)


def get_undersample_data2(data):
    number_records_buy = len(data[data['target'] == 0])
    buy_indices = np.array(data[data['target'] == 1].index)
    not_indices = data[data['target'] == 0].index

    random_not_indices = np.random.choice(buy_indices, number_records_buy, replace=False)
    random_not_indices = np.array(random_not_indices)

    under_sample_indices = np.concatenate([not_indices, random_not_indices])
    under_sample_data = data.iloc[under_sample_indices, :]

    # len(under_sample_data[under_sample_data.Class == 1]), len(under_sample_data[under_sample_data.Class == 0])
    X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'target']
    y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'target']
    print('undersampling: ', X_undersample.shape, y_undersample.shape)
    return under_sample_data

def get_undersample_data(data):
    number_records_buy = len(data[data['target'] == 1])
    buy_indices = np.array(data[data['target'] == 1].index)
    not_indices = data[data['target'] == 0].index

    random_not_indices = np.random.choice(not_indices, number_records_buy, replace=False)
    random_not_indices = np.array(random_not_indices)

    under_sample_indices = np.concatenate([buy_indices, random_not_indices])
    under_sample_data = data.iloc[under_sample_indices, :]

    # len(under_sample_data[under_sample_data.Class == 1]), len(under_sample_data[under_sample_data.Class == 0])
    X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'target']
    y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'target']
    print('undersampling: ', X_undersample.shape, y_undersample.shape)
    return under_sample_data

def get_X_y(under_sample_data):
    X_undersample = under_sample_data.loc[:, under_sample_data.columns != 'target']
    y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'target']
    return X_undersample, y_undersample

def load_train_test_split(train, n=5):
    splits = []
    for seed in params.TRAIN_TEST_SPLITS[:n]:
        X_train, X_test, y_train, y_test = train_test_split(train.loc[:, train.columns != 'target'], train['target'],
                                                            test_size=0.2, random_state=seed)
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        splits.append((X_train, X_test, y_train, y_test))
    return splits

