from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import numpy as np
import pandas as pd
import pipeline as pl

# loading the data
filename = 'data.pkl'
data = pl.read_data(filename)
# CENSUS_DATA_COLS is in download.py - maybe we should make it a standalone json for loading?
data = data.drop(list(CENSUS_DATA_COLS.values()), axis=1)

# defining independent and dependent variables
features = data.drop(['year', 'lat', 'lon', 'commuting_ridership'], axis=1)
target = data['commuting_ridership'].to_frame('commuting_ridership')
features = features._get_numeric_values()
features[features < 0] = np.nan

# splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1234)


# imputing null values with median
columns = ['median_income']
x_train, replacement = pl.impute(x_train, columns)
x_test, replacement = pl.impute(x_test, columns, replacement=replacement)

# defining models to use
scale = StandardScaler()
pca = PCA()
regr = LinearRegression()
lasso = Lasso(fit_intercept=True)
ridge = Ridge(fit_intercept=True)
en = ElasticNet(fit_intercept=True)
pf = PolynomialFeatures()

pipelines = {'regr': Pipeline([('scale', scale),
                               ('pf', pf),
                               ('pca', pca),
                               ('regr', regr)]),
            'lasso': Pipeline([('scale', scale),
                               ('pf', pf),
                               ('pca', pca),
                               ('lasso', lasso)]),
            'ridge': Pipeline([('scale', scale),
                               ('pf', pf),
                               ('pca', pca),
                               ('ridge', ridge)]),
            'elasticnet': Pipeline([('scale', scale),
                                    ('pf', pf),
                                    ('pca', pca),
                                    ('en', en)])}

# hyperparameter tuning (this number of parameters would take a while, so it would be
# best to use smaller ranges for testing)
# I used ranges of 2-5 for PCA and changed num to 3 for alphas, and it only took a 
# minute or so
params = {'regr': {'pca__n_components': np.arange(2, x_train.shape[1])},
          'lasso': {'pca__n_components': np.arange(2, x_train.shape[1]),
                    'lasso__alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
          'ridge': {'pca__n_components': np.arange(2, x_train.shape[1]),
                    'ridge__alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
          'elasticnet': {'pca__n_components': np.arange(2, x_train.shape[1]),
                         'en__alpha': [0.0001, 0.001, 0.01, 0.1, 1]}}

results = pd.DataFrame(columns=['params', 'mean_test_r2', 'mean_test_explained_variance', 
                                'mean_test_neg_mean_squared_error'])
for model, pipeline in pipelines.items():
    grid_search = GridSearchCV(estimator=pipeline, 
                  param_grid=params[model],
                  scoring=['r2', 'explained_variance', 'neg_mean_squared_error'],
                  cv=10,
                  refit='r2')
    model_result = grid_search.fit(x_train, y_train)
    df = pd.DataFrame(model_result.cv_results_)
    results = results.append(df[['params', 'mean_test_r2', 
                                 'mean_test_explained_variance', 
                                 'mean_test_neg_mean_squared_error']])