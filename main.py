'''
File to pull data, find best model, and recommend tracts for intervention
'''

import argparse
import pandas as pd
import model_selection as select
import download as dl
import model_selection
import recommend as rcmd

DATA_DF = pd.read_pickle("./pickle_files/final_data.pkl")
BEST_MODEL = pd.read_pickle('best_model.pkl').steps[-1][1]
POLY_DEGREE = pd.read_pickle('best_model.pkl').steps[-2][1].powers_[-1][-1]
K = 5

def go():
    '''
    Download data, find best model, and recommend tracts for intervention
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
        data_df = DATA_DF
        best_model = BEST_MODEL
        poly_degree = POLY_DEGREE

    elif args.d:
        print('Using archived files instead of pulling all data via API')
        data_df = DATA_DF
        pf, best_model = select.model_selection(K, data_df)

        poly_degree = pf.powers_[-1][-1]

    else:
        data_df = dl.compile_and_merge_data()
        pf, best_model = select.model_selection(K, data_df)

        poly_degree = pf.powers_[-1][-1]

    return rcmd.recommend_tracts_for_action(data_df, best_model,
                                                  num_poly=poly_degree)


if __name__ == '__main__':
    go()
