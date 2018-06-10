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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

# from util import get_X_y
from cii import get_prediction_dist
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import RandomOverSampler, SMOTE

cols = ['user_id', 'item_id']

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

def train_predict(models, names, X_train, y_train, X_test, y_test):

    # train = get_data([2,3,4])

    # train = util.get_undersample_data(train)
    # train = train[train.columns.difference(cols)]
    # test = test[test.columns.difference(cols)]
    # (X_train, y_train), (X_test, y_test) = util.get_X_y(train), util.get_X_y(test)
    #
    # print('before variance', X_train.shape)
    # vt = VarianceThreshold(threshold=44)
    # X_train = vt.fit_transform(X_train)
    # X_test = vt.transform(X_test)
    # print('after variance', X_train.shape)
    #
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_train = min_max_scaler.fit_transform(X_train)
    # X_test = min_max_scaler.fit_transform(X_test)


    # pca = PCA(n_components=30)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)

    statics_arr = []
    scoring = {'recall': 'recall', 'accuracy': 'accuracy', 'auc': 'roc_auc'}  # , 'auc': 'roc_auc'

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

            print('test precision: %f roc_auc: %f' % (test_prec, test_auc))

            static = [name, test_auc, test_prec, elapsed_time, *get_prediction_dist(y_test, test_pred), model]
            statics_arr.append(static)

        except Exception as ex:
            print(ex)

    statics = pd.DataFrame(statics_arr, columns=['Model', 'Test_auc', 'Test_precision', 'Time', 'Correct', 'Error',
                                                 'Correct value counts', 'Error value counts', 'Estimator'])
    statics.sort_values(by=['Test_auc'], ascending=False, inplace=True)

    statics.to_csv(params.output + '_'.join(['1452983', '2b', 'statics']) + '.txt', index=False)
    return np.array(statics_arr)[:, -1]

def get_models():
    nb = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=3)
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    rf = RandomForestClassifier(random_state=6, class_weight=None, max_features='log2', n_estimators=30)
    bagging = BaggingClassifier()
    dt = DecisionTreeClassifier(random_state=6, class_weight='balanced', max_features='auto')
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

def get_X_y(data):
    X = data.loc[:, data.columns != 'target']
    y = data.loc[:, data.columns == 'target']
    return X, y

def feature_selection(X, y):
    print(X.shape)
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, threshold='0.7*mean', prefit=True)
    return model


if __name__ == '__main__':
    train_flag = 1

    train = get_data(params.train_path + '/234' + params.b_train_file_name)
    test = get_data(params.train_path + '/456' + params.b_train_file_name)
    predicted = get_data(params.train_path + '/567' + params.b_train_file_name)
    train = train.drop(cols, axis=1)
    test = test.drop(cols, axis=1)

    # train = util.get_undersample_data(train)
    # print('undersample', train.shape)
    (X_train, y_train), (X_test, y_test) = get_X_y(train), get_X_y(test)

    print(X_train.shape, y_train.shape)
    X_train, y_train = RandomOverSampler('minority').fit_sample(X_train, y_train)
    print(X_train.shape, y_train.shape)

    fs_model = feature_selection(X_train, y_train)
    print(X_train.shape)
    X_train = fs_model.transform(X_train)
    X_test = fs_model.transform(X_test)
    X_predict = fs_model.transform(predicted.drop(cols, axis=1))
    print(X_train.shape)

    if train_flag:
        models, names = get_models()
        estimators = train_predict(models, names, X_train, y_train, X_test, y_test)
        for estimator, name in zip(estimators, names):
            util.save_to_file(predicted[cols], estimator.predict(X_predict),
                              '_'.join(['1452983', '2b', name]) + '.txt')

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




