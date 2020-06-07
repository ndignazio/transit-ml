'''
Main file to pull data and and find best model
'''

import argparse
import pandas as pd
import model_selection as select
import download

DATA_DF = 'data.pkl'

def go():
    '''
    Downlaod data and find best model
    '''
    parser = argparse.ArgumentParser(description='Get data and find best model')
    parser.add_argument('-a',
                        default=False,
                        action='store_true',
                        help='Use archived version of data')

    args = parser.parse_args()
    if args.a:
        print('Using archived files instead of pulling all data via API')
        data_df = DATA_DF
    else:
        data_df = download.compile_and_merge_data()

    ## This function doesn't exist yet
    best_model = model_selection.find_best_model()

    ## What other functions do we need? What should our software be returning?
    ## The best model + some characteristics of that model?

if __name__ == '__main__':
    go()
