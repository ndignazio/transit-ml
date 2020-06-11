from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import pipeline as pl
from math import sqrt
from ast import literal_eval
import datetime
import json
from pipeline import grid_search_cv, find_best_model, run_best_model, format_keynames
import warnings
warnings.filterwarnings("ignore")

# defining models to use
scale = StandardScaler()
regr = LinearRegression()
lasso = Lasso(max_iter=5000)
ridge = Ridge(max_iter=5000)
en = ElasticNet(max_iter=5000)
dt = DecisionTreeRegressor()
rf = RandomForestRegressor(random_state=0, n_jobs=-1, verbose=2)
pf = PolynomialFeatures()

PIPELINES = {"regr": Pipeline([("scale", scale),
                               ("pf", pf),
                               ("regr", regr)]),
            "lasso": Pipeline([("scale", scale),
                               ("pf", pf),
                               ("lasso", lasso)]),
            "ridge": Pipeline([("scale", scale),
                               ("pf", pf),
                               ("ridge", ridge)]),
            "elasticnet": Pipeline([("scale", scale),
                                    ("pf", pf),
                                    ("elasticnet", en)]),
            "decisiontree": Pipeline([("pf", pf),
                                      ("decisiontree", dt)]),
            "randomforest": Pipeline([("pf", pf),
                                      ("randomforest", rf)])}


PARAMS = {
'regr': {'pf__degree': [1, 2]},
'lasso': {'pf__degree': [1, 2],
          'lasso__alpha': [0.0001, 0.001, 0.01, 0.1]},
'ridge': {'pf__degree': [1, 2],
          'ridge__alpha': [0.0001, 0.001, 0.01, 0.1]},
'elasticnet': {'pf__degree': [1, 2],
               'elasticnet__alpha': [0.0001, 0.001, 0.01, 0.1]},
'decisiontree': {'pf__degree': [1, 2],
                 'decisiontree__criterion': ["mse", "friedman_mse", "mae"],
                 'decisiontree__max_depth': [5, 10, 15, 20],
                 'decisiontree__min_samples_split': [2, 5, 10]},
'randomforest': {'pf__degree': [1, 2],
                 'randomforest__criterion': ['mse', 'mae'],
                 'randomforest__n_estimators': [100, 200, 300],
                 'randomforest__max_depth': [5, 10, 15]}}


PARAMS_SMALL = {
'regr': {'pf__degree': [1, 2]},
'lasso': {'pf__degree': [1, 2],
          'lasso__alpha': [0.0001, 0.001, 0.01]}
}

PIPELINES_SMALL = {'regr': Pipeline([('scale', scale),
                               ('pf', pf),
                               ('regr', regr)]),
            'lasso': Pipeline([('scale', scale),
                               ('pf', pf),
                               ('lasso', lasso)])}


def model_selection(k, df, small=True, verbose=False):
    '''
    Selects best model given preselected models and hyperparameters.
    Runs smaller model for testing if small is True.
    Inputs: k (int) specification of number of folds for k-fold cross-
    validation
    df (DataFrame) pre-cleaned data
    small (boolean) a flag indicating whether the user wants to use a smaller
    pipeline for testing or the larger pipeline
    verbose (boolean) a flag indicating whether the user wants to see formatted
    output of the model in addition to return values
    Returns: (tuple) polynomial features and model steps of Pipeline object for the best
    model
    '''
    data = df

    features = data.drop(['commuting_ridership', 'GEO_ID'], axis=1)
    target = data['commuting_ridership'].to_frame('commuting_ridership')

    # splitting data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

    # imputing median for 1 value in median_income column
    columns = ['median_income']
    x_train, replacement = pl.impute(x_train, columns)
    x_test, replacement = pl.impute(x_test, columns, replacement=replacement)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    if small: 
        pipelines = PIPELINES_SMALL
        params = PARAMS_SMALL
    else:
        pipelines = PIPELINES
        params = PARAMS

    best, results = grid_search_cv(pipelines, params, 
                                   'neg_root_mean_squared_error', k, 
                                   x_train, y_train)
    now = datetime.datetime.now()
    filename = 'pickle_files/grid_search_results_{}.pkl'.format(str(datetime.datetime.now()))
    results.to_pickle(filename)
    (model, best_params), score = find_best_model(best)
    results, df, best_model = run_best_model(pipelines, model, best_params, 
                                             x_train, y_train, x_test, y_test)
    
    # If verbose is True, print information about model selection, parameters,
    # evaluation metrics, and feature importances
    if verbose:

        cv_params = format_keynames(best_params)
        cv_params['Model'] = model
        cv_params['Score'] = '{0:.3f}'.format(-score)

        print('---------------------------------')
        print('Best Model From Cross Validation')
        print('---------------------------------')
        for key, value in cv_params.items():
            print(key + ': ' + str(value))
        
        print('------------------------------------------------')
        print('Results of Running Best Model on Entire Dataset')
        print('------------------------------------------------')
        for key, value in results.items():
            print(key + ': ' + str(value))
        print('-------------------------------------------')
        print('Feature Importances of the Best Model')
        print('-------------------------------------------')
        print(df)

    return best_model.named_steps["pf"], best_model.named_steps[model]
