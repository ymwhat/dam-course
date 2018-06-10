import pandas as pd
import numpy as np
import time
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_validate, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from util import get_X_y
import util, params
from tuning import tuning
from cii import get_prediction_dist
cols = ['user_id']

def get_data(train_file_name):
    try:
        print("reading data file ", os.path.abspath(train_file_name))
        train = pd.read_csv(train_file_name)
        return train
    except:
        print("data not found, creating them..")
    finally:
        print('reading data finished..')


def train_predict(models, names, X_train, y_train, X_test, y_test):
    statics_arr = []
    scoring = {'recall': 'recall', 'accuracy': 'accuracy',  'auc': 'roc_auc'}  # , 'auc': 'roc_auc'

    for model, name in zip(models, names):
        kfold = StratifiedKFold(n_splits=3, random_state=78)
        print('=' * 30)
        try:
            cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
            for score in scoring:
                msg = "%s: %s mean: %f " % (name, score, cv_results['test_' + score].mean())
                print(msg)

            start_time = time.time()
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            test_prec = precision_score(y_test, test_pred, average='micro')
            test_auc = roc_auc_score(y_test, test_pred)
            elapsed_time = time.time() - start_time

            print('Test precision: %f roc_auc: %f' % (test_prec, test_auc))

            static = [name, test_auc, test_prec, elapsed_time, *get_prediction_dist(y_test, test_pred), model]
            statics_arr.append(static)

        except Exception as ex:
            print(ex)

    statics = pd.DataFrame(statics_arr, columns=['Model', 'Test_auc', 'Test_precision', 'Time', 'Correct', 'Error', 'Correct value counts', 'Error value counts', 'Estimator'])
    statics.sort_values(by=['Test_auc'], ascending=False, inplace=True)
    statics.to_csv(params.output + '_'.join(['1452983', '2ci', 'statics']) + '.txt', index=False)
    return np.array(statics_arr)[:, -1]

def get_models():
    nb = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=1)
    ada = AdaBoostClassifier(n_estimators=50)
    rf = RandomForestClassifier(random_state=6, class_weight='balanced',  n_estimators=35)
    bagging = BaggingClassifier()
    dt = DecisionTreeClassifier(random_state=6, class_weight='balanced')
    lr_l1 = LogisticRegression(C=0.3, penalty='l1')
    gbdt = GradientBoostingClassifier()

    name = ['RandomForestClassifier',
            'GradientBoostingClassifier',
            'BaggingClassifier',
            'AdaBoostClassifier',
            'DecisionTreeClassifier',
            'GaussianNB',
            'KNeighborsClassifier',
            'LR_l1']
    model = [rf, gbdt, bagging, ada, dt, nb, knn]

    return model, name


def feature_selection(X, y):
    print(X.shape)
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True, threshold='median')
    return model

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

if __name__ == '__main__':
    train_flag = 1

    train = get_data(params.train_path + '/234' + params.ci_train_file_name)
    test = get_data(params.train_path + '/456' + params.ci_train_file_name)
    predicted = get_data(params.train_path + '/567' + params.ci_train_file_name)

    train = train.drop(cols, axis=1)
    test = test.drop(cols, axis=1)
    X_predict = predicted.drop(cols, axis=1)

    (X_train, y_train), (X_test, y_test) = get_X_y(train), get_X_y(test)

    # X_train, y_train = RandomOverSampler('minority').fit_sample(X_train, y_train)
    # X_train, y_train, _ = RandomUnderSampler(return_indices=True).fit_sample(X_train, y_train)

    fs_model = feature_selection(X_train, y_train)
    print(X_train.shape, y_train.shape)
    X_train = fs_model.transform(X_train)
    print(X_train.shape, y_train.shape)
    X_test = fs_model.transform(X_test)
    X_predict = fs_model.transform(predicted.drop(cols, axis=1))

    if train_flag:
        models, names = get_models()
        estimators = train_predict(models, names, X_train, y_train, X_test, y_test)
        for estimator, name in zip(estimators, names):
            util.save_to_file(predicted[cols], estimator.predict(X_predict),
                              '_'.join(['1452983', '2ci', name]) + '.txt')

    else:
        model_params = {
            'LogisticRegression': (LogisticRegression(penalty='l2'), {
                # 'penalty':['l2', 'l1'], l2没用
                # 'C': np.arange(0.2, 0.5, 0.1)
                'C': np.arange(0.2, 0.5, 0.1)
            }),
            'DecisionTreeClassifier': (DecisionTreeClassifier(), {
                'class_weight': ['balanced', None]
            }),
            'KNeighborsClassifier': (KNeighborsClassifier(), {
                'n_neighbors' : range(2, 5, 1)
            })
        }
        rf = (RandomForestClassifier(class_weight=None, max_features='log2', random_state=6), {
            # 'class_weight': ['balanced', None],
            # 'max_features' : ['auto', 'log2', 'sqrt', None],
            'n_estimators': range(5, 35, 5)
        })
        dt = (DecisionTreeClassifier(random_state=6,), {
            'class_weight': ['balanced', None],
            'max_features': ['auto', 'log2', 'sqrt', None]
        })
        # tuning(train, *dt, scoring='roc_auc', unsample=False)
        # tuning(train, *rf, scoring='roc_auc', unsample=False)


