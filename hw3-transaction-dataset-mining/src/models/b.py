import params, util
from tuning import tuning
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import model_selection
import os
from sklearn import preprocessing
from build_feature import Feature
import build_feature
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer

def read_raw_data(self, file_name='../../data/raw/trade_new.csv'):
    data = pd.read_csv(file_name, index_col=0)
    if data.columns.contains('sldatime'):
        data = data.rename(columns={'sldatime': 'sldat'})
    data['sldat'] = pd.to_datetime(data['sldat'], errors='coerce')
    data['bndno'] = data['bndno'].fillna(-1).astype('int', errors='ignore')
    data['pluno'] = data['pluno'].astype('int', errors='ignore')
    data['dptno'] = data['dptno'].astype('int', errors='ignore')
    data['month'] = data['sldat'].apply(lambda x: x.month)
    data['day'] = data['sldat'].apply(lambda x: x.day)
    data = data.rename(columns={'pluno': 'item_id', 'dptno': 'cat_id', 'bndno': 'brand_id', 'vipno': 'user_id'})
    print("read data finished")
    return data


def get_data(train_file_name):
    try:
        print("reading data..", os.path.abspath(train_file_name))
        train = pd.read_csv(train_file_name)

    except:
        print("data not found, creating them in build_feature.py")
    finally:
        print('reading data finished..')
        return train

def train(models):
    train = get_data(params.train_path + '/234'+params.b_train_file_name)
    test = get_data(params.train_path + '/456'+params.b_train_file_name)
    print(train.shape)
    predicted = get_data(params.train_path + '/567' + params.b_train_file_name)

    cols=['user_id', 'item_id']
    # train = get_data([2,3,4])

    train = util.get_undersample_data(train)
    train = train[train.columns.difference(cols)]
    test = test[test.columns.difference(cols)]
    (X_train, y_train), (X_test, y_test) = util.get_X_y(train), util.get_X_y(test)

    print('before variance', X_train.shape)
    vt = VarianceThreshold(threshold=44)
    X_train = vt.fit_transform(X_train)
    X_test = vt.transform(X_test)
    print('after variance', X_train.shape)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)


    # pca = PCA(n_components=30)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)

    results = []
    names = []
    elapsed = []
    auc = []
    precision = []
    corrects, errors = [], []
    corrects_value_counts, errors_value_counts = [], []

    scoring = {'recall': 'recall', 'accuracy': 'accuracy'}#, 'auc': 'roc_auc'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=3, random_state=78)
        print('='*30)
        try:
            cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)

            for score in scoring:
                msg = "%s: %s mean:%f std:(%f)" % (name, score, cv_results['test_'+score].mean(), cv_results['test_'+score].std())
                print(msg)

            start_time = time.time()

            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)

            # test_pred = model.predict(X_test[X_test.columns.difference(['user_id', 'item_id'])])
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

            print('correct: {} errors: {}'.format(len(correct),len(error)))
            print('predict correct value counts:')
            print(y_test['target'].iloc[correct].value_counts())
            print('predict error value counts:')
            print(y_test['target'].iloc[error].value_counts())


            # util.save_to_file(X_test[['user_id', 'item_id']], test_pred,
            #              '_'.join(['1452983', '2b', name]) + '.txt')

        except Exception as ex:
            print(ex)

    statics = pd.DataFrame(
        [names, auc, precision, elapsed, corrects, errors, corrects_value_counts, errors_value_counts]).T
    statics.columns = ['name', 'auc', 'precision', 'time', 'correct', 'error', 'corrects_value_counts',
                       'errors_value_counts']
    statics.sort_values(by=['auc'], ascending=False, inplace=True)

    statics.to_csv(params.output + '_'.join(['1452983', '2b', 'statics']) + '.txt', index=False)
    print('end')


if __name__ == '__main__':
    train_flag = 1
    if train_flag:
        models = []
        # models.append(('LR', LogisticRegression(C=0.7, class_weight='balanced', penalty='l2')))
        # models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('RF', RandomForestClassifier()))
        # models.append(('LR_l1', LogisticRegression(C=0.8, penalty='l1')))
        models.append(('Bagging', BaggingClassifier()))
        models.append(('Adaboost', AdaBoostClassifier()))
        models.append(('GBDT', GradientBoostingClassifier()))
        # models.append(('SVM', svm.SVC()))
        train(models)
    else:
        model_params = {
            'LogisticRegression': (LogisticRegression(penalty='l1'), {
                # 'penalty': ['l2', 'l1'], l2貌似没用
                'C': np.arange(0.4, 1, 0.2)#l2:score不变
            }),
            'DecisionTreeClassifier': (DecisionTreeClassifier(), {
                'class_weight': ['balanced', None]

            })
        }
        train, _ = get_data(params.ci_train_file_name)
        tuning(train, model_params['LogisticRegression'][0], model_params['LogisticRegression'][1], scoring='roc_auc')
        # tuning(model_params['LogisticRegression'][0], model_params['LogisticRegression'][1], scoring='accuracy')
        # tuning(model_params['LogisticRegression'][0], model_params['LogisticRegression'][1], scoring='roc_auc', unsample=False)




