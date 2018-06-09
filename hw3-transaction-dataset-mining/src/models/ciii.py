import util, params
from tuning import tuning
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_validate
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer
def get_data(train_file_name):
    try:
        print("reading data file ", os.path.abspath(train_file_name))
        train = pd.read_csv(train_file_name)
        return train
    except:
        print("data not found, creating them..")
    finally:
        print('reading data finished..')


def train(models):
    test = get_data(params.train_path + '/456' + params.ciii_train_file_name)
    train = get_data(params.train_path + '/234' + params.ciii_train_file_name)
    predicted = get_data(params.train_path + '/567' + params.ciii_train_file_name)


    cols = ['user_id', 'cat_id']
    train = train[train.columns.difference(cols)]
    test = test[test.columns.difference(cols)]

    # train = util.get_undersample_data(train)
    (X_train, y_train), (X_test, y_test) = util.get_X_y(train), util.get_X_y(test)

    print(X_train.shape)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # normalizer = Normalizer().fit(X_train)
    # X_train = normalizer.transform(X_train)
    # X_test = normalizer.transform(X_test)
    # vt = VarianceThreshold(threshold=3)
    # X_train = vt.fit_transform(X_train)
    # SelectKBest(lambda X_train, y_train: np.array(map(lambda x: pearsonr(x, y_train), X_train.T)).T, k=2).fit_transform(X_train, y_train)
    # sk = SelectKBest(mutual_info_classif, k=2)
    # X_train = sk.fit_transform(X_train, y_train)
    # X_test = sk.transform(X_test)

    print(X_train.shape)

    results = []
    names = []
    elapsed = []
    auc = []
    precision = []
    corrects, errors = [], []
    corrects_value_counts, errors_value_counts = [], []

    scoring = {'recall': 'recall', 'accuracy': 'accuracy'}  # , 'auc': 'roc_auc'
    for name, model in models:
        kfold = KFold(n_splits=3, random_state=78)
        print('=' * 30)
        try:
            cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)

            for score in scoring:
                msg = "%s: %s mean:%f std:(%f)" % (
                name, score, cv_results['test_' + score].mean(), cv_results['test_' + score].std())
                print(msg)
            start_time = time.time()

            model.fit(X_train, y_train)

            # model.fit(X_train[X_train.columns.difference(cols)], y_train)
            # test_pred = model.predict(X_test[X_test.columns.difference(cols)])

            test_pred = model.predict(X_test)
            test_prec = precision_score(y_test, test_pred, average='micro')
            test_auc = roc_auc_score(y_test, test_pred)
            print('test precision: %f roc_auc: %f' % (test_prec, test_auc))

            elapsed_time = time.time() - start_time

            results.append(cv_results)
            auc.append(test_auc)
            precision.append(test_prec)
            elapsed.append(elapsed_time)
            names.append(name)

            correct = np.where(np.array(y_test['target'].tolist()) == test_pred)[0]
            error = np.where(np.array(y_test['target'].tolist()) != test_pred)[0]

            corrects.append(len(correct))
            errors.append(len(error))
            corrects_value_counts.append(y_test['target'].iloc[correct].value_counts().to_dict())
            errors_value_counts.append(y_test['target'].iloc[error].value_counts().to_dict())

            print('correct: {} errors: {}'.format(len(correct), len(error)))
            print('predict correct value counts:')
            print(y_test['target'].iloc[correct].value_counts())
            print('predict error value counts:')
            print(y_test['target'].iloc[error].value_counts())


            # pre = model.predict(predicted)
            # util.save_to_file(predicted[cols], pre,
            #                   '_'.join(['1452983', '2cii', name]) + '.txt')

        except Exception as ex:
            print(ex)

    statics = pd.DataFrame(
        [names, auc, precision, elapsed, corrects, errors, corrects_value_counts, errors_value_counts]).T
    statics.columns = ['name', 'auc', 'precision', 'time', 'correct', 'error', 'corrects_value_counts',
                       'errors_value_counts']
    statics.sort_values(by=['auc'], ascending=False, inplace=True)
    statics.to_csv(params.output + '_'.join(['1452983', '2ciii', 'statics']) + '.txt', index=False)
    print('end')

if __name__ == '__main__':
    train_flag = 1
    if train_flag:
        models = []

        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('RF', RandomForestClassifier()))
        models.append(('Bagging', BaggingClassifier()))
        models.append(('Adaboost', AdaBoostClassifier()))
        models.append(('GBDT', GradientBoostingClassifier()))
        models.append(('LR', LogisticRegression()))
        # models.append(('LR_l1', LogisticRegression(C=0.5, penalty='l1')))
        train(models)
    else:
        model_params = {
            'LogisticRegression': (LogisticRegression(penalty='l1'), {'C': np.arange(0.5, 0.6, 0.1)})
        }
        train, _ = get_data(params.ci_train_file_name)
        tuning(train, model_params['LogisticRegression'][0], model_params['LogisticRegression'][1], scoring='accuracy')






