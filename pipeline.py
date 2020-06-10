import pandas as pd
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
from ast import literal_eval
from math import sqrt
import datetime
import censusdata
import pickle

def get_acs_5_data(year, state, data_aliases):
    '''
    Get American Community Survey 5-year data at block group level

    Inputs:
        year (integer): year from which to pull data
        state (string): encoding of state for which to pull data
        data_aliases (dictionary; keys and values both strings): mapping of
            encoded data columns to pull from ACS with their descriptive names.
            Note that these descriptive names will be the headers of the output
            DataFrame. See below links for 2018 column encodings:
            https://api.census.gov/data/2018/acs/acs5/variables.html
            https://api.census.gov/data/2018/acs/acs1/variables.html
            https://api.census.gov/data/2018/acs/acsse/variables.html

    (For more information on Census geographies, please visit this link:
        https://www.census.gov/data/developers/geography.html)

    Output:
        A pandas dataframe with ACS data
    '''
    # Initialize dataframe
    if data_aliases:
        results_df = pd.DataFrame(columns=data_aliases.values())
    else:
        results_df = pd.DataFrame(columns=data_columns)

    # print("Data columns are...", data_aliases.keys())

    results_df['year'] = ""

    # Get Census data and load into dataframe
    geographies = censusdata.geographies(censusdata.censusgeo([('state', state),
        ('county', '*')]), 'acs5', year)

    # i = 0
    for v in list(geographies.values()):
        ( (_, _) , (_, county_code) ) = v.params()
        # print("County code is...", county_code)
        df = censusdata.download("acs5", year, censusdata.censusgeo(
            [("state", state), ("county", county_code), ("tract", "*")]),
            list(data_aliases.keys()), key="e62f1cebce1c8d3afece25fc491fbec7271a588b").reset_index()
        # print("On loop...", i)

    # for year in years:
    #     df = censusdata.download(survey_type, year,
    #                             censusdata.censusgeo([("state", state),
    #                                                   (subgroup_str[survey_type],
    #                                                   subgroup_var[survey_type])]),
    #                                                   data_columns)
        # if data_aliases:
        df = df.rename(columns=data_aliases)
        df['year'] = year

        results_df = results_df.append(df, ignore_index=True)
        # i+=1

    results_df = results_df.infer_objects()

    return results_df


def read_data(filename):
    '''
    Reads files into dataframes or geodataframes, depending on suffix.
    Input: filename (str)
    Output: a pandas dataframe or a geopandas geodataframe
    '''
    ending = filename.split('.')[-1]
    data = None

    if ending == 'csv':
       data = pd.read_csv(filename)

    elif ending == 'pkl':
        with open(filename, 'rb') as f:
            data = pickle.load(f)

    elif ending == 'geojson':
        data = gpd.read_file(filename)

    return data


def explore_df(df):
    '''
    Get summary stats and sample of dataframe, by calling
        explore_df_summary_stats and explore_df_sample functions
    '''
    explore_df_summary_stats(df)
    print("--------------------------------------------------------------")
    explore_df_sample(df)


def explore_df_summary_stats(df):
    '''
    Given a dataframe, print its shape and various summary statistics

    Inputs:
        df: A pandas dataframe

    Outputs:
        - Sentence stating # of rows and columns in dataframe
        - A dataframe with summary statistics for all quantitative columns
        - A listing of all columns with NA or Null values (and how many)
        - A listing of all columns with negative minimum values (and what
            that value is)
    '''
    rows, columns = df.shape
    print("The dataframe has {:,} rows and {:,} columns.".format(rows,
                                                                 columns))
    print("--------------------------------------------------------------")

    df_stats = df.describe()
    print("Detailed descriptive statistics of quantitative columns:")
    display(df_stats)

    print("--------------------------------------------------------------")

    cols_with_null = []
    print("Quantitative columns with null/NA values:")
    for col in df_stats.columns:
        num_null = rows - df_stats[col]['count']
        if num_null > 0:
            cols_with_null.append(col)
            print("\nColumn: {}".format(col))
            print("Number of null/NA values: {}".format(num_null))

    print("--------------------------------------------------------------")

    print("Quantitative columns with negative minimum values:")
    for col in df_stats.columns:
        min_val = df_stats[col]['min']
        if min_val < 0:
            print("\nColumn: {}".format(col))
            print("Min value: {:,}".format(min_val))

    return cols_with_null


def explore_df_sample(df):
    '''
    Given a dataframe, print the types of each column, along with several
        rows of the actual dataframe

    Input:
        df: A pandas dataframe

    Output:
        A pandas Series with the type of each column, and a smaller version
            of the input dataframe (with the first 5 rows)
    '''
    print("Column types:\n")
    print(df.dtypes)

    print("--------------------------------------------------------------")

    print("First five rows of dataframe:")
    display(df.head())


def impute(df, columns, replacement=None):
    '''
    Imputes missing values in continuous variables with the sample
    median of that variable (taken from the training set).
    Inputs: df (DataFrame)
    columns (list): column names
    Output: Nothing; input df is modified
    '''
    filtered = df[columns]
    if replacement is None:
        guide = {}
        filtered = filtered.apply(lambda x: x.fillna(x.median(), axis=0))
        for column in columns:
            guide[column] = filtered[column].median()
        replacement = guide

    else:
        filtered = filtered.fillna(value=replacement, axis=0)

    df[columns] = filtered

    return df, replacement


def grid_search_cv(pipelines, params, scoring, cv, x_train, y_train):
    '''
    Runs cross validation on multiple sklearn Pipeline objects.
    Inputs: pipelines (dict) dictionary of pipeline objects
    params (dict) parameters corresponding to pipeline objects
    scoring (str) chosen evaluation metric name
    cv (int) number of folds to use in cross-validation
    x_train (DataFrame) training features
    y_train (DataFrame) training targets
    Returns: best (dict) a dictionary with keys as model names and values as
    tuples with the first tuple entry as a dictionary of model parameters
    results (DataFrame) a table of model parameters and mean test score results
    '''

    start = datetime.datetime.now()
    # initializing empty results dataframe
    results = pd.DataFrame(columns=['params',
                                    'mean_test_score'])
    # initializing empty best dictionary
    best = {}
    # looping through pipelines
    for model, pipeline in pipelines.items():
        # performing grid search on each pipeline
        print('Running Grid Search on {} with the following parameters: {}'.format(model, params[model]))
        grid_search = GridSearchCV(estimator=pipeline, 
                      param_grid=params[model],
                      scoring=scoring,
                      cv=cv,
                      refit=scoring)
        # fitting model to data
        model_result = grid_search.fit(x_train, y_train)
        # adding entry to best dictionary for the best model in pipeline
        best[(model, str(grid_search.best_params_))] = grid_search.best_score_
        # create dataframe of cross-validation results and store parameters
        # and mean test scores
        df = pd.DataFrame(model_result.cv_results_)
        results = results.append(df[['params',
                                     'mean_test_score']])

    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    return best, results

def find_best_model(best, neg=True):
    '''
    select best model out of dictionary of models, parameters, and scores
    based on best score.
    Input: best (dict) a dictionary with keys as model names and values as
    tuples with the first tuple entry as a dictionary of model parameters 
    and the second tuple entry as the mean test score of the model.
    Returns: choice (tuple) the best model, parameters, and score
    '''
    model = None
    choice = max(best.values())
    if neg:
        choice = min(best.values())

    if len(best) > 1:

        for model_params, score in best.items():
            if neg:
                if score > choice:
                    choice = score
                    model = model_params
                    continue
            elif score > choice:
                choice = score
                model = model_params
    else:
        model = list(best.keys())[0]
        choice = best[model]

    return model, choice

def format_keynames(params):
    '''
    Format keynames from parameter dictionary of sklearn Pipeline.
    Input: params (dict) parameters formatted like so: 
           '<named_step>__<parameter>'
    Returns: params (dict) parameter dictionary with '<named_step>__'
    removed
    '''
    if not isinstance(params, dict):
        params = literal_eval(params)
    for key in params.keys():
        new_key = key.split("_")[-1]
        params[new_key] = params.pop(key)
    return params


def run_best_model(pipelines, mod, params, x_train, y_train, x_test, y_test):
    #{'pf__degree': 2, 'randomforest__criterion': 'mae', 'randomforest__n_estimators': 300, 'randomforest__max_depth': 15}
    '''
    Runs the best model selected from the find_best_model function.
    Inputs: pipelines (dict) dictionary of pipeline objects
    mod (str) type of model corresponding to pipelines dictionary
    params (dict) parameters of the best model
    x_train, y_train, x_test, y_test (4 Pandas DataFrames) training and testing 
    data
    Returns: tuple
    first element: (dict) results of running the model on entire dataset
    second element: (DataFrame) nonzero feature importances of model sorted in
    ascending order
    '''
    best_model = pipelines[mod]
    best_model.set_params(**literal_eval(params))
    best_model.fit(x_train, y_train)
    pkl_filename = "best_model.pkl"
    with open(pkl_filename, 'wb') as f:
        pickle.dump(best_model, f)
    predictions = best_model.predict(x_test)
    metrics = {}     
    metrics['Model'] = mod                                   
    metrics['RMSE'] = '{0:.3f}'.format(sqrt(mean_squared_error(y_test, predictions))) 
    metrics['R2'] = '{0:.3f}'.format(r2_score(y_test, predictions))
    tuples = []
    feature_names = best_model.named_steps['pf'].get_feature_names(x_train.columns)
    if mod == 'randomforest' or mod == 'decisiontree':
        for i, name in enumerate(feature_names):
            tuples.append((name, best_model.named_steps[mod].feature_importances_[i]))
    else:
        for i, name in enumerate(feature_names):
            tuples.append((name, best_model.named_steps[mod].coef_[i]))

    sorted_by_coef = sorted(tuples, key=lambda tup: abs(tup[1]), reverse=True)
    df = pd.DataFrame(sorted_by_coef, columns=['label', 'coefficient'])
    df = df[df['coefficient'] != 0]
    params = format_keynames(params)
    return {**metrics, **params}, df, best_model
