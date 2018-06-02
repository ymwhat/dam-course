import util, params
from tuning import tuning
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
import os
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor


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
    train, test = get_data(params.civ_train_file_name, params.civ_test_file_name, train_month=[2, 3, 4],
                                test_month=[5])

    # train = util.get_undersample_data2(train)
    train.drop('user_id', inplace=True, axis=1)
    (X_train, y_train), (X_test, y_test) = util.get_X_y(train), util.get_X_y(test)

    print(test.shape)
    print(train.shape)

    results = []
    names = []
    elapsed = []
    mae, evs, mse = [],[],[]


    scoring = {'neg_mean_absolute_error': 'neg_mean_absolute_error', 'explained_variance': 'explained_variance', 'r2' : 'r2'}  # , 'mae': 'roc_mae'
    for name, model in models:
        kfold = KFold(n_splits=5, random_state=78)
        print('=' * 30)
        try:
            cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)

            for score in scoring:
                msg = "%s: %s mean:(%f) std:(%f)" % (name, score, cv_results['test_' + score].mean(), cv_results['test_' + score].std())
                print(msg)

            start_time = time.time()

            model.fit(X_train, y_train)
            test_pred = model.predict(X_test.loc[:, X_test.columns != 'user_id'])
            test_pred = np.array([0 if x < 0 else x for x in [x for x in test_pred]])
            test_evs, test_mae, test_mse = explained_variance_score(y_test, test_pred), mean_absolute_error(y_test, test_pred), mean_squared_error(y_test, test_pred)

            print('test explained_variance_score: %f mean_absolute_error: %f mean_squared_error: %f' % (test_evs, test_mae, test_mse))

            elapsed_time = time.time() - start_time

            results.append(cv_results)
            mae.append(test_mae)
            evs.append(test_evs)
            mse.append(test_mse)
            elapsed.append(elapsed_time)
            names.append(name)
            output = pd.DataFrame(X_test['user_id'])
            output['target'] = pd.Series(test_pred.reshape(-1))
            output['label'] = y_test
            output.to_csv(params.output + '_'.join(['1452983', '2civ', name]) + '.txt', index=False, header=False)

        except Exception as ex:
            print('exception ', ex)

    statics = pd.DataFrame([names, mae, evs, mse, elapsed, [x[1] for x in models]]).T
    statics.columns=['name', 'mae', 'evs', 'mse', 'time', 'estimator']
    statics.sort_values(by=['mse'], ascending=True, inplace=True)
    statics.to_csv(params.output + '_'.join(['1452983', '2civ', 'statics']) + '.txt', index=False)
    print('end')

if __name__ == '__main__':
    train_flag = 1
    if train_flag:
        models = []
        # models.append(('LR', LogisticRegression(penalty='l1', C=0.9, class_weight='balanced')))
        # models.append(('LR_l1', LogisticRegression(C=0.3, penalty='l1')))
        # , max_depth = 2, max_leaf_nodes = 300,
        models.append(('CART', DecisionTreeRegressor(splitter='best', max_features='log2', max_depth=5, random_state=12)))
        models.append(('LinearRegression', LinearRegression(fit_intercept=True)))
        models.append(('Ridge', Ridge(fit_intercept=False, alpha=1)))
        models.append(('Lasso', Lasso(fit_intercept=True, alpha=0.5)))
        models.append(('BaggingRegressor', BaggingRegressor(
            base_estimator=DecisionTreeRegressor(splitter='random', max_depth=7),
            n_estimators=55, max_features=0.4, bootstrap=True)))
        models.append(('RandomForestRegressor', RandomForestRegressor(n_estimators=20, max_features=None, max_leaf_nodes=400)))
        models.append(('AdaBoostRegressor', AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(splitter='random', max_depth=2),
            n_estimators=60)))

        train(models)

    else:
        model_params = {
            'LinearRegression' : (LinearRegression(), {
                'fit_intercept' : [True, False]
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

        # tuning(train, *model_params['CART'], scoring='mean_squared_error', unsample=False)
        # tuning(train, *model_params['BaggingRegressor'], scoring='neg_mean_squared_log_error', unsample=False)
        # tuning(train, *model_params['RandomForestRegressor'], scoring='neg_mean_squared_log_error', unsample=False)
        # tuning(train, *model_params['AdaBoostRegressor'], scoring='neg_mean_squared_error', unsample=False)

