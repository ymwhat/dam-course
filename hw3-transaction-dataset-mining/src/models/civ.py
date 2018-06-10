import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.gaussian_process import GaussianProcessRegressor
import util, params
from tuning import tuning

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
    cols = ['user_id']
    ci_train = get_data(params.train_path + '/234' + params.ci_train_file_name)
    ci_test = get_data(params.train_path + '/456' + params.ci_train_file_name)

    (ci_X_train, ci_y_train) = util.get_X_y(ci_train)
    (ci_X_test, ci_y_test) = util.get_X_y(ci_test)

    # rf = RandomForestClassifier(random_state=6, class_weight='balanced',  n_estimators=35)
    rf = GaussianNB()
    rf.fit(ci_X_train.drop(cols, axis=1), ci_y_train)

    # scoring = {'neg_mean_absolute_error': 'neg_mean_absolute_error', 'explained_variance': 'explained_variance', 'r2' : 'r2'}
    for model, name in zip(models, names):
        # kfold = StratifiedKFold(n_splits=3, random_state=78)
        print('=' * 30)
        print(name)
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)

            buy_or_not = rf.predict(ci_X_test.drop(cols, axis=1))
            tmp = np.array(list(zip(test_pred, buy_or_not)))
            test_pred = tmp[:, 1] * tmp[:, 0]

            elapsed_time = time.time() - start_time
            test_evs, test_mae, test_mse = explained_variance_score(y_test, test_pred), \
                                           mean_absolute_error(y_test, test_pred), \
                                           mean_squared_error(y_test, test_pred)

            print('Test explained_variance_score: %f mean_absolute_error: %f mean_squared_error: %f' % (test_evs, test_mae, test_mse))

            static = [name, test_evs, test_mae, test_mse, elapsed_time, model]
            statics_arr.append(static)
            output = pd.DataFrame()
            output['target'] = y_test
            output['label'] = test_pred
            output = output.round(2)
            output.to_csv(params.output + '_'.join(['1452983', '2civ-test', name]) + '.txt', index=False)

        except Exception as ex:
            print(ex)

    statics = pd.DataFrame(statics_arr, columns=['Model', 'Test_EVS', 'Test_MAE', 'Test_MSE', 'Time',  'Estimator'])
    statics.sort_values(by=['Test_MSE'], ascending=True, inplace=True)
    statics.to_csv(params.output + '_'.join(['1452983', '2civ', 'statics']) + '.txt', index=False)
    return np.array(statics_arr)[:, -1]


def get_models():
    ada = AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(splitter='random', max_depth=2),
            n_estimators=60)
    rf = RandomForestRegressor(n_estimators=20, max_features=None, max_leaf_nodes=400)
    bagging = BaggingRegressor(
            base_estimator=DecisionTreeRegressor(splitter='random', max_depth=7),
            n_estimators=55, max_features=0.4, bootstrap=True)
    dt = DecisionTreeRegressor(splitter='best', max_features='log2', max_depth=5, random_state=12)
    lasso = Lasso(fit_intercept=True, alpha=0.5)
    # ada = AdaBoostRegressor()
    # rf = RandomForestRegressor(n_estimators=20, max_features=None, max_leaf_nodes=400)
    # bagging = BaggingRegressor()
    # dt = DecisionTreeRegressor()

    gbdt = GradientBoostingRegressor()
    knn = KNeighborsRegressor()
    gpr = GaussianProcessRegressor()
    name = ['RandomForestRegressor',
            'BaggingRegressor',
            'AdaBoostRegressor',
            'DecisionTreeRegressor',
            'GradientBoostingRegressor',
            'KNeighborsRegressor',
            'GaussianProcessRegressor'
            ]
    model = [rf, bagging, ada, dt, gbdt, knn, gpr]

    return model, name


if __name__ == '__main__':
    train_flag = 1
    cols = ['user_id']
    train = get_data(params.train_path + '/234' + params.civ_train_file_name)
    test = get_data(params.train_path + '/456' + params.civ_train_file_name)
    predicted = get_data(params.train_path + '/567' + params.civ_train_file_name)

    train = train.drop(cols, axis=1)
    test = test.drop(cols, axis=1)
    X_predict = predicted.drop(cols, axis=1)
    (X_train, y_train), (X_test, y_test) = util.get_X_y(train), util.get_X_y(test)

    if train_flag:
        models, names = get_models()
        estimators = train_predict(models, names, X_train, y_train, X_test, y_test)
        for estimator, name in zip(estimators, names):
            output = pd.DataFrame(predicted['user_id'])
            output['label'] = estimator.predict(X_predict)
            output = output.round(2)
            output.to_csv(params.output + '_'.join(['1452983', '2civ', name]) + '.txt', index=False, header=False)
    else:
        model_params = {
            'LinearRegression': (LinearRegression(), {
                'fit_intercept': [True, False]
            }),
            'Ridge': (Ridge(), {
                'alpha': np.arange(0.5, 10.0, 0.5),
                'fit_intercept': [True, False]
            }),
            'Lasso': (Lasso(), {
                'alpha': np.arange(0.5, 5.0, 0.3),
                'fit_intercept': [True, False],
                # positive=True
            }),
            'CART': (DecisionTreeRegressor(splitter='random', max_features='log2', max_depth=7, random_state=0), {
                # 'splitter': ['best', 'random'],
                # 'max_depth': list(range(5, 8, 1)) + [None],
                'max_leaf_nodes': [None] + list(range(100, 1000, 100))
                # 'max_features': [None, 'log2', 'sqrt']
            }),
            'BaggingRegressor':(BaggingRegressor(base_estimator=DecisionTreeRegressor(splitter='random', max_depth=7),
                                                 n_estimators=55, max_features=0.4, bootstrap=True), {
                # 'base_estimator__max_depth': list(range(2, 15, 3)) + [None],
                # 'base_estimator__splitter': ['best', 'random'],
                # 'n_estimators': range(45, 70, 5),
                # 'max_features': np.arange(0.4, 1.0, 0.2),
                # 'bootstrap': [True, False]
            }),
            'RandomForestRegressor':(RandomForestRegressor(n_estimators=20, max_features=None, max_leaf_nodes=400), {
                # 'n_estimators': range(5, 30, 5),
                # 'max_features':  [None, 'log2', 'sqrt'],
                'max_leaf_nodes': [None]+list(range(400, 800, 50))
            }),
            'AdaBoostRegressor': (AdaBoostRegressor(base_estimator=DecisionTreeRegressor(splitter='random', max_depth=2),
                                                    n_estimators=60), {
                'base_estimator__max_depth': list(range(2, 15, 3)) + [None],
                # 'base_estimator__splitter': ['best', 'random'],
                'n_estimators': range(10, 70, 5)
            })
        }
        train, _ = get_data(params.ci_train_file_name, params.ci_test_file_name, train_month=[2, 3, 4],
                                 test_month=[5])
        # tuning(train, *model_params['LinearRegression'], scoring='neg_mean_squared_error', unsample=False)
        # tuning(train, *model_params['Ridge'], scoring='neg_mean_squared_error', unsample=False)
        tuning(train, *model_params['Lasso'], scoring='neg_mean_squared_error', unsample=False)
