import pandas as pd
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

import datetime
import censusdata
import pickle

# Config: Dictionaries of models and hyperparameters
MODELS = {
    'LogisticRegression': LogisticRegression(),
    'LinearSVC': LinearSVC(),
    'GaussianNB': GaussianNB()
}

GRID = {
    'LogisticRegression': [{'penalty': x, 'C': y, 'random_state': 0, 'max_iter': 5000}
                           for x in ('l2', 'none') \
                           for y in (0.01, 0.1, 1, 10, 100)],
    'GaussianNB': [{'priors': None}],
    'LinearSVC': [{'C': x, 'random_state': 0, 'max_iter': 5000} \
                  for x in (0.01, 0.1, 1, 10, 100)]
}

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

def parse(column):
    """
    Parses dates into datetime format using a dictionary lookup.
    Inputs: column (column of a Pandas dataframe)
    Returns: the column parsed into datetime format
    """
    dates = {}
    for date in column.unique():
        dates[date] = pd.to_datetime(date)

    return column.map(dates)

def describe(df):
    '''
    Generates heatmap overlaid on correlation table of dataframe object.
    Input: df (DataFrame)
    Output: correlation table
    '''
    return df.describe()

def dist(column):
    '''
    Generates distribution of values in a column.
    Input: column (Series)
    Output: value counts (Series)
    '''
    return column.value_counts()

def correlate(df):
    '''
    Generates correlation table of variables in a DataFrame.
    Input: df (DataFrame)
    Output: correlation table
    '''
    return df.corr()

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


def train_test(df, predictors, target, test_size=0.20, random_seed=1234):
    '''
    Splits data into training and testing sets.
    Inputs: df (DataFrame)
    predictors (list): list of predictor variables
    target (list): list of target variables
    test_size (float): the portion of the data that will be assigned to
    the test set
    random_seed (int): random seed for reproducibility
    '''

    x_train, x_test, y_train, y_test = train_test_split(predictors, target,
                                                        test_size=test_size,
                                                        random_state=random_seed)

    return (x_train, x_test, y_train, y_test)

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


def standardize(df, columns, scaler=None):
    '''
    If scaler is not none, use given scaler's means and sds to standardize (used for test set case)
    '''

    # Normalizing train set
    filtered = df[columns]
    if scaler is None:
      scaler = StandardScaler()
      normalized_features = scaler.fit_transform(filtered)
      df[columns] = normalized_features

    # Normalizing test set (with the values based on the training set)
    else:
      normalized_features = scaler.transform(filtered)
      df[columns] = normalized_features

    return df, scaler

def encode(x_train, x_test, column):
    '''
    Performs one-hot encoding of categorical variable, with necessary
    adjustments between training and test sets. The original column is
    dropped.
        1) If there is a value in the training set that is not in the
        test set, we initialize a column of zeros in the test set.
        2) If there is a value in the test set that is not in the
        training set, we remove the record containing this value from
        the test set.
    Inputs: x_train (DataFrame): the training set
    x_test (DataFrame): the test set
    column (str): the name of the column to be encoded
    Outputs: train, test (Dataframes): updated, one-hot encoded training
    and test sets
    '''

    if isinstance(x_train, pd.Series):
        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)

    for val in x_test[column].unique():
        if val not in x_train[column].unique():
            x_test = x_test[x_test[column] != val]

    train_dummies = pd.get_dummies(x_train[column])
    test_dummies = pd.get_dummies(x_test[column])
    train_dummies.columns = train_dummies.columns.astype(str)
    test_dummies.columns = test_dummies.columns.astype(str)

    for col in train_dummies:
        if col not in test_dummies:
            test_dummies[col] = 0

    train = x_train.drop(column, axis=1).assign(**train_dummies)
    test = x_test.drop(column, axis=1).assign(**test_dummies)

    return train, test

def encoder(x_train, x_test, columns):
    '''
    Performs one-hot encoding on categorical data. If a value in the test set
    is not in the training set, the value is removed from the test set. If
    a value in the training set is not in the test set, a column of zeros is
    added to the test set with the appropriate label.
    Input: x_train (Series or DataFrame): training dataset
    x_test (Series or DataFrame): testing dataset
    column (str): the name of the column to be encoded
    Output: train, test: transformed DataFrame objects
    '''

    for col in columns:
        x_train, x_test = encode(x_train, x_test, col)

    return x_train, x_test

def discretize(series, bounds, labels):
    '''
    Bins continuous data into discrete chunks.
    Inputs: series (Series): a Series or column of a Pandas DataFrame
    bounds (list): specification of boundaries on which to cut
    labels (list): labels for the bins, from lowest to highest magnitude
    Output: series: the discretized series
    '''
    bins = len(labels)
    series_copy = series.copy()
    series_copy = pd.cut(series_copy, bounds, labels=labels)
    return series

def fit_predict(x_train, x_test, y_train, model):

    model.fit(x_train, y_train)
    return model.predict(x_test)


def grid_search(x_train, x_test, y_train, y_test, models=MODELS, params=GRID):
    '''
    Performs grid search of models with selected parameters.
    Inputs: x_train (df): training predictors
    x_test (df): testing predictors
    y_train (df): training target
    y_test (df): testing target
    models (dict): learning models
    params (dict): parameters for each model
    Output: results (df) a summary of results with accuracy, precision, and
    recall scores.
    '''
    start = datetime.datetime.now()

    # Convert target variables to ndarray
    y_test = np.ravel(y_test)
    y_train = np.ravel(y_train)

    # Initialize results data frame
    results = pd.DataFrame(columns=['Model', 'Parameters', 'Accuracy', 'Precision', 'Recall'])

    # Loop over models
    for model_key in MODELS.keys():

        # Loop over parameters
        for params in GRID[model_key]:
            print("Training model:", model_key, "|", params)

            # Create model
            model = MODELS[model_key]
            model.set_params(**params)

            # Predict on testing set
            predictions = fit_predict(x_train, x_test, y_train, model)

            # Evaluate predictions
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)

            # Store results in your results data frame
            results = results.append({'Model': model_key, 'Parameters': params, 'Accuracy': accuracy,
                                      'Precision': precision, 'Recall': recall}, ignore_index=True)

    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)

    return results
