'''
Main file to pull data and and find best model
'''

import argparse
import pandas as pd
import model_selection as select
import download
import model_selection

DATA_DF = pd.read_pickle("./pickle_files/final_data.pkl")
BEST_MODEL = pd.read_pickle('best_model.pkl').steps[1][1]
POLY_DEGREE = pd.read_pickle('best_model.pkl').steps[0][1].powers_[-1][-1]


def go():
    '''
    Downlaod data and find best model
    '''
    parser = argparse.ArgumentParser(description="""Get data, find best model,
    and recommend tracts for review""")
    parser.add_argument('-d',
                        default=False,
                        action='store_true',
                        help='Use archived version of data')

    parser.add_argument('-m',
                        default=False,
                        action='store_true',
                        help='Use archived version of model and data')

    args = parser.parse_args()

    if args.m:
        print('''Using archived model and data instead of pulling data via
        API and finding best model using grid search''')
        best_model = BEST_MODEL
    elif args.d:
        print('Using archived files instead of pulling all data via API')
        data_df = DATA_DF
    else:
        data_df = download.compile_and_merge_data()

    ## This function doesn't exist yet
    features = data.drop(['commuting_ridership'], axis=1)
    target = data['commuting_ridership'].to_frame('commuting_ridership')
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
    best_model_pipe = model_selection.run_model_selection(5, x_train, y_train, x_test, y_test)

    return None

    ## What other functions do we need? What should our software be returning?
    ## The best model + some characteristics of that model?

if __name__ == '__main__':
    go()
