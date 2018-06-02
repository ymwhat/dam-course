import util, params
from tuning import tuning
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_validate

def get_data(train_file_name, test_file_name, train_month=[2, 3, 4], test_month=[5]):
    try:
        print("reading data file ", os.path.abspath(train_file_name), os.path.abspath(test_file_name))
        train = pd.read_csv(train_file_name)
        test = pd.read_csv(test_file_name)

    except:
        print("ci data not found, creating them..")
        train = util.Feature(month=train_month)
        train.user_data(train_file_name)

        test = util.Feature(month=test_month)
        test.user_data(test_file_name)

        train = pd.read_csv(train_file_name)
        test = pd.read_csv(test_file_name)
    finally:
        print('reading data finished..')
        return train, test

def train(models):
    train, test = get_data(params.ci_train_file_name, params.ci_test_file_name, train_month=[2, 3, 4],
                                test_month=[5])
    print(test['target'].value_counts())
    print(train['target'].value_counts())
    # train = util.get_undersample_data2(train)
    train.drop('user_id', inplace=True, axis=1)
    (X_train, y_train), (X_test, y_test) = util.get_X_y(train), util.get_X_y(test)

    print(test['target'].value_counts())
    print(train['target'].value_counts())

    results = []
    names = []
    elapsed = []
    auc = []
    precision = []
    corrects, errors=[], []
    corrects_value_counts, errors_value_counts=[],[]

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
            test_pred = model.predict(X_test.loc[:, X_test.columns != 'user_id'])
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


            util.save_to_file(X_test[['user_id']], test_pred,
                               '_'.join(['1452983', '2ci', name]) + '.txt')

        except Exception as ex:
            print(ex)

    statics = pd.DataFrame([names, auc, precision, elapsed, corrects, errors, corrects_value_counts, errors_value_counts]).T
    statics.columns=['name', 'auc', 'precision', 'time', 'correct', 'error', 'corrects_value_counts', 'errors_value_counts']
    statics.sort_values(by=['auc'], ascending=False, inplace=True)
    statics.to_csv(params.output + '_'.join(['1452983', '2ci', 'statics']) + '.txt', index=False)
    print('end')

if __name__ == '__main__':
    train_flag = 0
    if train_flag:
        models = []
        models.append(('LR', LogisticRegression(penalty='l1', C=0.9, class_weight='balanced')))
        # models.append(('LR_l1', LogisticRegression(C=0.3, penalty='l1')))

        # models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier(class_weight='balanced')))
        models.append(('NB', GaussianNB()))
        models.append(('RF', RandomForestClassifier()))

        train(models)

    else:
        model_params = {
            'LogisticRegression': (LogisticRegression(penalty='l1', class_weight='balanced'), {
                # 'penalty':['l2', 'l1'], l2没用 0.5
                # 'C': np.arange(0.2, 0.5, 0.1)
                'C': np.arange(0.5, 1, 0.2),
                # 'class_weight': ['balanced', None]

            }),
            'DecisionTreeClassifier': (DecisionTreeClassifier(), {
                'class_weight': ['balanced', None]

            })
        }
        train, _ = get_data(params.ci_train_file_name, params.ci_test_file_name, train_month=[2, 3, 4],
                                 test_month=[5])
        # train = util.get_undersample_data2(train)
        tuning(train, *model_params['LogisticRegression'], scoring= 'roc_auc', unsample=False)
        # tuning(train, *model_params['DecisionTreeClassifier'], scoring= 'roc_auc', unsample=False)






