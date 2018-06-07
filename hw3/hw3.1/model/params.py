TRAIN_TEST_SPLITS = [ 4966, 58023, 19639, 63847, 30619, 58170, 19854, 37524, 37938, 9618]
NUM_FOLDS = 3

ROW = 65
COL = 82


BASE_DT_PARAMS = {
    'criterion': 'gini',
    'class_weight': 'balanced',
    'max_features': None,
    'splitter': 'best',

    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0006
}
BASE_KNN_PARAMS = {
    'n_neighbors': 1,
    'metric': 'euclidean',
    'weights': 'uniform'
}

BASE_ADABOOST_PARAMS = {
    'n_estimators': 50,
    'algorithm': 'SAMME.R',
    # 'learning_rate': 1.5,
    'class_weight' : 'balanced',
    'max_features': None,
    'splitter': 'best',#
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0008
}

BASE_BAGGING_PARAMS = {
    'n_estimators' : 40,
    'max_samples' : 1.0,
    'max_features': 0.88,

    'class_weight' : 'balanced',
    'dt_max_features': None,
    'splitter': 'best',#
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0006
}

BASE_RF_PARAMS = {
    'n_estimators': 35,
    'max_features': None,
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.00045
}


BASE_GB_PARAMS = {
    'n_estimators': 35,
    'max_depth': None,
    'criterion': 'friedman_mse',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'subsample': 1,
    # 'learning_rate': 1.5,
    'max_features': None,
    'max_leaf_nodes': None,
    # 'min_impurity_decrease': 0.0008
}

train_path = '../data/interim/train'
train_file = train_path + '/train.pkl'
raw_path = '../data/raw'
gongcan = raw_path + '/2g_gongcan.csv'
data_2g = raw_path + '/data_2g.csv'

pkl_path = '../data/interim'
train_pkl = pkl_path+ '/train.pkl'
output = '../data/output/'
sfolds = pkl_path + '/sfolds.pkl'
top_pkl = '../data/interim/top.pkl'
dap_pkl = '../data/interim/dap.pkl'
errors_pkl = '../data/interim/errors.pkl'
b_errors_pkl = '../data/interim/b_errors.pkl'
c_errors_pkl = '../data/interim/c_errors.pkl'
d_errors_pkl = '../data/interim/d_errors.pkl'
e_errors_pkl = '../data/interim/e_errors.pkl'

