import util, params
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from util import get_X_y, get_undersample_data
def tuning(train, model, params_dist, scoring='roc_auc', unsample=True):
    if unsample:
        train = get_undersample_data(train)
    X_train, y_train = get_X_y(train)
    kfold = StratifiedKFold(n_splits=4, random_state=78)
    grid_search = GridSearchCV(estimator=model, param_grid=params_dist, scoring=scoring, n_jobs=8, cv=kfold,
                               refit=False, verbose=2)

    grid_search.fit(X_train, y_train)

    print(grid_search.best_score_, grid_search.best_params_)
    print(grid_search.cv_results_)

