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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_validate

def get_data(train_file_name, test_file_name):
    try:
        print("reading data file ", os.path.abspath(train_file_name), os.path.abspath(test_file_name))
        train = pd.read_csv(train_file_name)
        test = pd.read_csv(test_file_name)
    except:
        print("ci data not found, creating them in build_feature.py")
    finally:
        print('reading data finished..')
        return train, test

def train(models):
    train, test = get_data(params.train_path + '/456'+params.ci_train_file_name, params.test_path + '/7'+params.ci_test_file_name)
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
        kfold = KFold(n_splits=5, random_state=78)
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

    statics = pd.DataFrame([names, auc, precision, elapsed, corrects, errors, corrects_value_counts, errors_value_counts, [x[1] for x in models]]).T
    statics.columns=['name', 'auc', 'precision', 'time', 'correct_number', 'error_number', 'corrects_value_counts', 'errors_value_counts', 'estimator']
    statics.sort_values(by=['auc'], ascending=False, inplace=True)
    statics.to_csv(params.output +'_'.join(['1452983', '2ci', 'statics']) + '_456' + '.txt', index=False)
    print('end')

if __name__ == '__main__':
    train_flag = 1
    if train_flag:
        models = []
        # models.append(('LR', LogisticRegression(penalty='l1', C=0.9, class_weight='balanced')))
        # models.append(('LR_l1', LogisticRegression(C=0.3, penalty='l1')))
        models.append(('CART', DecisionTreeClassifier(class_weight=None, splitter='best', max_features='log2', max_depth=2, max_leaf_nodes=300, random_state=0)))
        models.append(('NB', GaussianNB()))
        models.append(('RF', RandomForestClassifier(n_estimators=10, max_features='log2', max_leaf_nodes=300,  max_depth=2, random_state=0)))
        models.append(('KNN', KNeighborsClassifier(n_neighbors=7, weights='uniform', metric='manhattan', leaf_size=20)))
        models.append(('Adaboost', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(class_weight='balanced', splitter='best', max_features='sqrt') , n_estimators=30,  random_state=0)))
        models.append(('BaggingClassifier-knn',BaggingClassifier(
            base_estimator=KNeighborsClassifier(n_neighbors=10, metric='manhattan', weights='uniform', leaf_size=21),
            n_estimators=20, max_features=0.4, bootstrap=False, random_state=0)))

        models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))

        train(models)
    else:
        model_params = {
            'LogisticRegression': (LogisticRegression(penalty='l1', class_weight='balanced'), {
                # 'penalty':['l2', 'l1'], l2没用 0.5
                # 'C': np.arange(0.2, 0.5, 0.1)
                'C': np.arange(0.5, 1, 0.2),
                # 'class_weight': ['balanced', None]
            }),
            'DecisionTreeClassifier': (DecisionTreeClassifier(class_weight='balanced'), {
                # 'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', None],
                'splitter': ['best', 'random'],
                'max_features': [None, 'sqrt', 'log2'],
                'max_depth': list(range(2, 8, 1))+[None],
                'max_leaf_nodes': range(100, 1000, 100)
            })
            ,'RandomForestClassifier': (RandomForestClassifier(oob_score=True, verbose=2, random_state=0), {
                'n_estimators': range(5, 30, 5),
                'criterion': ['gini', 'entropy'],
                'max_features': [None, 'sqrt', 'log2'],
                'max_depth': range(2, 20, 2)

            }),
            # KNeighborsClassifier doesn't support sample_weight.
            'AdaBoostClassifier':(AdaBoostClassifier(base_estimator=KNeighborsClassifier(), algorithm='SAMME'), {
                'n_estimators' : range(10, 50, 10)
            }),
            'KNeighborsClassifier': (KNeighborsClassifier(n_jobs=-1, weights='uniform'), {
                'n_neighbors': range(1, 8, 2),
                # 'weights': ['uniform', 'distance']
                # 'metric': ['manhattan', 'euclidean'],
                'leaf_size': range(20, 25, 1)
            }),
            'BaggingClassifier': (BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=10, metric='manhattan', weights='uniform', leaf_size = 21),
                                                    n_estimators=20, max_features=0.4, bootstrap=False, random_state=0), {
                # 'base_estimator__n_neighbors': range(2, 15, 2),
                # 'base_estimator__leaf_size': range(15, 30, 2),
                'n_estimators': range(8, 30, 2),
                'max_features': np.arange(0.4, 1.0, 0.2),
                'bootstrap':[True, False]
            }),
            'BaggingClassifier-dt': (BaggingClassifier(
                base_estimator=DecisionTreeClassifier(),
                n_estimators=20, max_features=0.4, bootstrap=False, random_state=0), {
                                      # 'base_estimator__n_neighbors': range(2, 15, 2),
                                      # 'base_estimator__leaf_size': range(15, 30, 2),
                                      'n_estimators': range(8, 30, 2),
                                      'max_features': np.arange(0.4, 1.0, 0.2),
                                      'bootstrap': [True, False]
                                  }),
            'GradientBoostingClassifier':(GradientBoostingClassifier(), {

            })
        }

        train, _ = get_data(params.ci_train_file_name, params.ci_test_file_name)
        # train = util.get_undersample_data2(train)
        # tuning(train, *model_params['LogisticRegression'], scoring= 'roc_auc', unsample=False)
        # tuning(train, *model_params['DecisionTreeClassifier'], scoring='roc_auc', unsample=False)
        # tuning(train, *model_params['KNeighborsClassifier'], scoring='roc_auc', unsample=False, drop_col=['user_id'])
        # tuning(train, *model_params['RandomForestClassifier'], scoring='roc_auc', unsample=False)
        tuning(train, *model_params['AdaBoostClassifier'], scoring='roc_auc', unsample=False)
        # tuning(train, *model_params['BaggingClassifier'], scoring='roc_auc', unsample=False)
