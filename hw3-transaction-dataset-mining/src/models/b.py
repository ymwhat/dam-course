import params, util
from tuning import tuning
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import model_selection
import os
from build_feature import Feature
import build_feature
def get_data(train_file_name):
    try:
        print("reading data..", os.path.abspath(train_file_name))
        train = pd.read_csv(train_file_name)
        # f = Feature()
        # data = f.read_raw_data('../../data/raw/trade_new.csv')
        # f.data = data
        # f.m_action_cr = build_feature.monthly_action_cr(data, month, groupby=['user_id', 'item_id', ['user_id', 'item_id']],
        #               prefixes=['u_', 'i_', 'ui_'])
        # f.m_action_cr_agg = build_feature.monthly_action_cr_agg(f.m_action_cr, groupby=['user_id', 'item_id', ['user_id', 'item_id']],
        #               prefixes=['u_', 'i_', 'ui_'])
        #
        # f.users = f.user_profile()
        # f.items = f.item_profile()
        #
        # cols = ['user_id', 'item_id']
        # ui = f.user_join_item()
        # if label:
        #     ui = f.user_item_label(ui, f.month[-1] + 1, cols)
        #
        # ui = ui.merge(f.m_action_cr_agg[str(cols)], on=cols, how='left')
        # ui.fillna(0, inplace=True)
        # train = ui
        # ui.to_csv(file_name, index=False)
        # ui[['user_id', 'item_id', 'target']].to_csv(file_name + 'label', index=False)
        # print('user_item_data:', ui.columns)

    except:
        print("data not found, creating them in build_feature.py")
    finally:
        print('reading data finished..')
        return train

def train(models):
    train = get_data(params.train_path + '/234'+params.b_train_file_name)
    test = get_data(params.train_path + '/456'+params.b_train_file_name)
    cols=['user_id', 'item_id']
    # train = get_data([2,3,4])
    train = train[train.columns.difference(cols)]
    train = util.get_undersample_data(train)

    (X_train, y_train), (X_test, y_test) = util.get_X_y(train), util.get_X_y(test)

    print(train['target'].value_counts())

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
            test_pred = model.predict(X_test[X_test.columns.difference(['user_id', 'item_id'])])
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


            util.save_to_file(X_test[['user_id', 'item_id']], test_pred,
                         '_'.join(['1452983', '2b', name]) + '.txt')

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
        models.append(('LR_l1', LogisticRegression(C=0.8, penalty='l1')))
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




