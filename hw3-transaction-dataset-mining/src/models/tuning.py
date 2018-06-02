import util, params
from sklearn.model_selection import GridSearchCV

def tuning(train, model, params_dist, scoring, unsample=True, drop_col=['user_id']):
    if unsample:
        train = util.get_undersample_data(train)

    train.drop(drop_col, inplace=True, axis=1)
    X_train, y_train = util.get_X_y(train)

    grid_search = GridSearchCV(estimator=model, param_grid=params_dist, scoring=scoring, n_jobs=8, cv=5,
                               refit=False, verbose=2)

    grid_search.fit(X_train, y_train)

    print(grid_search.best_score_, grid_search.best_params_)
    print(grid_search.cv_results_)

