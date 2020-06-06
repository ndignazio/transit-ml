from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import pipeline as pl
from math import sqrt
from ast import literal_eval
import datetime
import json
from pipeline import grid_search_cv, find_best_model

# loading the data
filename = 'data.pkl'
data = pl.read_data(filename)
# dropping untransformed census columns
with open('CENSUS_DATA_COLS.json') as f:
    COLS = json.load(f)
data = data.drop(list(COLS.values()), axis=1)

# defining independent and dependent variables
features = data.drop(['year', 'lat', 'lon', 'commuting_ridership'], axis=1)
target = data['commuting_ridership'].to_frame('commuting_ridership')
features = features._get_numeric_data()
features[features < 0] = np.nan

# splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# dropping a few irrelevant columns
x_train = x_train.drop(['index_right', 'num_nearby_routes', 'num_bus_routes', 
                        'num_rail_routes', 'num_other_routes'], axis=1)
x_test = x_test.drop(['index_right', 'num_nearby_routes', 'num_bus_routes', 
                      'num_rail_routes', 'num_other_routes'], axis=1)

# imputing null values with median
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
pf = PolynomialFeatures()


'''
pipelines = {'regr': Pipeline([('scale', scale),
                               ('pf', pf),
                               ('regr', regr)]),
            'lasso': Pipeline([('scale', scale),
                               ('pf', pf),
                               ('lasso', lasso)]),
            'ridge': Pipeline([('scale', scale),
                               ('pf', pf),
                               ('ridge', ridge)]),
            'elasticnet': Pipeline([('scale', scale),
                                    ('pf', pf),
                                    ('en', en)])}



hyperparameter tuning
since data is standardized, no coefficients are greater than abs(1)
therefore, alpha levels are between 0 and 1

params = {
'regr': {'pf__degree': [1, 2, 3]},
'lasso': {'pf__degree': [1, 2, 3],
          'lasso__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5]},
'ridge': {'pf__degree': [1, 2, 3],
          'ridge__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5]},
'elasticnet': {'pf__degree': [1, 2, 3],
               'en__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5]}}
'''
params = {
'regr': {'pf__degree': [1, 2,]},
'lasso': {'pf__degree': [1, 2],
          'lasso__alpha': [0.0001, 0.001, 0.01]}
}

pipelines = {'regr': Pipeline([('scale', scale),
                               ('pf', pf),
                               ('regr', regr)]),
            'lasso': Pipeline([('scale', scale),
                               ('pf', pf),
                               ('lasso', lasso)])}


if __name__ == '__main__':

    best, results = grid_search_cv(pipelines, params, 'neg_root_mean_squared_error', 5, x_train, y_train)
    (model, params), score = find_best_model(best)
    print('Best Model: {} with the following parameters: {} and a mean test score of {}'.format(model, params, score))
    best_model_params = literal_eval(params)
    alpha = None
    for key, val in params.items():
        if key.split('_')[0] in list(pipelines.keys()):
            alpha = val


    # run best model on whole dataset
    pf = PolynomialFeatures(degree=best_model_params['pf__degree'], include_bias=False)
    lasso = Lasso(alpha=alpha, max_iter=10000)
    model = Pipeline([('scale', scale),
                       ('pf', pf),
                       ('lasso', lasso)])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    root_mse = sqrt(mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error: ' + str(root_mse))
    # 0.0515480138425091
    normalized_root_mse = root_mse / (max(y_test) - min(y_test))
    print('Normalized RMSE: ' + str(normalized_root_mse))
    # 0.08043680214805388
    r2_score = r2_score(y_test, predictions)
    print('R2: ' + str(r2_score))
    # 0.8723402541626002
