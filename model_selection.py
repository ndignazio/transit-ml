from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import pandas as pd
import pipeline as pl

# loading the data
filename = 'merged_data.pkl'
data = pl.read_data(filename)
data = data.drop(list(CENSUS_DATA_COLS.values()), axis=1)

# defining independent and dependent variables
features = data.drop(['year', 'lat', 'lon', 'commuting_ridership'], axis=1)
target = data['commuting_ridership'].to_frame('commuting_ridership')

# splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1234)

# imputing null values with median
columns = x_train.columns
x_train, replacement = pl.impute(x_train, columns)
x_test, replacement = pl.impute(x_test, columns, replacement=replacement)

y_train, replacement = pl.impute(y_train, ['commuting_ridership'])
y_test, replacement = pl.impute(y_test, ['commuting_ridership'], replacement=replacement)

# defining models to use
pca = PCA()
regr = LinearRegression()
lasso = Lasso(fit_intercept=True)
ridge = Ridge(fit_intercept=True)
en = ElasticNet(fit_intercept=True)
scale = StandardScaler()

pipeline = Pipeline(steps=[('scale', scale),
                           ('pca', pca),
                           ('regr', regr),
                           ('lasso', lasso),
                           ('ridge', ridge),
                           ('en', en)])

# hyperparameter tuning
params = {'pca__n_components': np.arange(2, x_train.shape[1]),
          'lasso__alpha': np.linspace(0, 2, num=11),
          'ridge__alpha': np.linspace(0, 2, num=11),
          'en__alpha': np.linspace(0, 2, num=11)}

# grid search
gs = GridSearchCV(estimator=pipeline, 
                  param_grid=params,
                  scoring=['r2', 'explained_variance', 'neg_mean_squared_error'],
                  cv=10,
                  refit='r2')
model_result = gs.fit(x_train, y_train)
results = pd.DataFrame(model_result.cv_results_)
print(gs.best_score_)
print(gs.best_params_)