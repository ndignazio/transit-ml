from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
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

## ******* NOTE FROM MIKE ****************
## I think any cleaning done below should be done in
## download.py (e.g., dropping unnecessary columns and NAs)

# loading the data
filename = 'data.pkl'
data = pl.read_data(filename)
# dropping untransformed census columns
with open('CENSUS_DATA_COLS.json') as f:
    DATA_COLS = json.load(f)
keys = [key for key in list(DATA_COLS.values()) if key != 'GEO_ID']
data = data._get_numeric_data()
data[data < 0] = np.nan
    
#Dropping irrelevant columns
data = data.drop(keys, axis=1)
data = data.drop(['year', 'lat', 'lon', 'index_right', 'num_nearby_routes', 'num_bus_routes',
            'num_rail_routes', 'num_other_routes'], axis=1)


# defining independent and dependent variables
features = data.drop(['commuting_ridership'], axis=1)
target = data['commuting_ridership'].to_frame('commuting_ridership')

# splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# imputing null values in median income with median income
columns = ['median_income']
x_train, replacement = pl.impute(x_train, columns)
x_test, replacement = pl.impute(x_test, columns, replacement=replacement)

# reshaping targets appropriately
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# defining models to use
scale = StandardScaler()
regr = LinearRegression()
lasso = Lasso(max_iter=5000)
ridge = Ridge(max_iter=5000)
en = ElasticNet(max_iter=5000)
rf = RandomForestRegressor(random_state=0)
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
            "randomforest": Pipeline([("pf", pf),
                                      ("randomforest", rf)])}


PARAMS = {
'regr': {'pf__degree': [1, 2]},
'lasso': {'pf__degree': [1, 2],
          'lasso__alpha': [0.0001, 0.001, 0.01, 0.1]},
'ridge': {'pf__degree': [1, 2],
          'ridge__alpha': [0.0001, 0.001, 0.01, 0.1]},
'elasticnet': {'pf__degree': [1, 2],
               'en__alpha': [0.0001, 0.001, 0.01, 0.1]},
'randomforest': {'pf__degree': [1, 2],
                 'randomforest__n_estimators': [100, 200, 500],
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


def run_model_selection(k, x_train, y_train, x_test, y_test, small=True):
    '''
    Selects best model given preselected models and hyperparameters.
    Runs smaller model for testing if small is True.
    Inputs: k (int) specification of number of folds for k-fold cross-
    validation
    x_train, y_train, x_test, y_test (DataFrames) training and testing data
    small (boolean) a flag indicating whether the user wants to use a smaller
    pipeline for testing or the larger pipeline
    Returns: Nothing. Prints grid search results and the results of running 
    the best model from grid search on the entire dataset, including
    evaluation metrics and feature importances
    '''
    if small: 
        pipelines = PIPELINES_SMALL
        params = PARAMS_SMALL
    else:
        pipelines = PIPELINES_SMALL
        params = PARAMS_SMALL
    best, results = grid_search_cv(pipelines, params, 'neg_root_mean_squared_error', k, x_train, y_train)
    (model, best_params), score = find_best_model(best)
    cv_params = format_keynames(params)
    cv_params['Model'] = model
    cv_params['Score'] = score
    print('---------------------------------')
    print('Best Model From Cross Validation')
    print('---------------------------------')
    print('Model: {}'.format(model))
    for key, value in literal_eval(best_params).items():
        print(key + ': ' + str(value))
    results, df = run_best_model(pipelines, model, best_params, x_train, y_train, x_test, y_test)
    print('------------------------------------------------')
    print('Results of Running Best Model on Entire Dataset')
    print('------------------------------------------------')
    for key, value in results.items():
        print(key + ': ' + str(value))
    print('-------------------------------------------')
    print('5 Most Important Features of the Best Model')
    print('-------------------------------------------')
    print(df)
    
