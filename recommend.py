'''
Recommend tracts for intervention consideration or review
'''
import json
import pandas as pd
import pipeline
from sklearn.preprocessing import PolynomialFeatures


def create_adjusted_features_df(features_df, tscore_num_to_add):
    '''
    Create new features dataframe, with transit score feature adjusted
        up by fixed amount
    '''
    max_transit_score = 100
    threshold = max_transit_score - tscore_num_to_add


    new_features = features_df.copy(deep=True)
    new_features['transit_score'] = new_features['transit_score'] + \
                                        tscore_num_to_add
    new_features['transit_score'][new_features['transit_score'] > \
                                    max_transit_score] = max_transit_score

    return new_features


def recommend_tracts_for_action(df, model_obj, n_tracts, tscore_addition=10,
                                num_poly=None):
    '''
    Recommend top N tracts intervention consideration. These tracts
        are the ones that our model predicts will see the largest increase
        in commuter transit ridership (measured by the % of commuters
        that use public transit to commute to work) of all tracts
        considered.

    Inputs:
        df: A pandas dataframe where each row is a census tract, and the
            columns are model features or is the model target
        model_obj: A fitted model object
        n_tracts (integer): Number of tracts to output
        tscore_addition (integer): Number to add to transit score to allow
            model to predict new target
        num_poly (integer): polynomial expansion to apply to the data

    Output:
        A pandas dataframe, where each row is a tract
    '''
    new_df = df.copy(deep=True)
    pipeline.impute(new_df, ['median_income'])

    features = new_df.drop(columns=['GEO_ID', 'commuting_ridership'], axis=1)
    # print("The number of columns in this df is...", len(features.columns.to_list()))
    # print("The columns are...", features.columns.to_list())
    features_new_tscore = create_adjusted_features_df(features, tscore_addition)

    if num_poly:
        poly = PolynomialFeatures(num_poly)
        features = pd.DataFrame(poly.fit_transform(features))
        features_new_tscore = pd.DataFrame(poly.fit_transform(features_new_tscore))

    current_state_predvals = model_obj.predict(features)
    new_tscore_predvals = model_obj.predict(features_new_tscore)

    new_df['pred_tgt_current'] = current_state_predvals
    new_df['pred_tgt_new_tscore'] = new_tscore_predvals
    new_df['pred_chg_commuting_ridership'] = new_df['pred_tgt_new_tscore'] \
                                                - new_df['pred_tgt_current']

    results_df = new_df.drop(columns=['pred_tgt_current',
                                        'pred_tgt_new_tscore'], axis=1)

    # Move predicted change and GEO_ID to front
    pred_chg = new_df['pred_chg_commuting_ridership']
    tract_id = new_df['GEO_ID'].str[9:]

    results_df.drop(columns=['GEO_ID', 'pred_chg_commuting_ridership'],
                    axis=1, inplace = True)
    results_df.insert(0, 'tract_id', tract_id)
    results_df.insert(2, 'pred_chg_commuting_ridership', pred_chg)

    return results_df.sort_values(by='pred_chg_commuting_ridership',
                                  ascending=False).head(30)


def recommend_tracts_for_review(df, model_obj, n_tracts, num_poly=None):
    '''
    Tracts with large negative gaps between model-predicted commuter
        transit ridership, and actual ridership

    Inputs:
        df: A pandas dataframe where each row is a census tract, and the
            columns are model features or is the model target
        model_obj: A fitted model object
        n_tracts (integer): Number of tracts to output

    Output:
        A pandas dataframe, where each row is a tract

    '''
    new_df = df.copy(deep=True)
    pipeline.impute(new_df, ['median_income'])

    features = new_df.drop(columns=['GEO_ID', 'commuting_ridership'], axis=1)

    if num_poly:
        poly = PolynomialFeatures(num_poly)
        features = pd.DataFrame(poly.fit_transform(features))

    current_state_predvals = model_obj.predict(features)
    new_df['model_pred_ridership'] = current_state_predvals
    new_df['diff_actual_and_model_pred'] = new_df['commuting_ridership'] \
                                                - new_df['model_pred_ridership']

    # Move difference, model-pred ridership GEO_ID to front
    model_pred = new_df['model_pred_ridership']
    diff = new_df['diff_actual_and_model_pred']
    tract_id = new_df['GEO_ID'].str[9:]

    new_df.drop(columns=['GEO_ID', 'model_pred_ridership',
                    'diff_actual_and_model_pred'], axis=1, inplace = True)
    new_df.insert(0, 'tract_id', tract_id)
    new_df.insert(1, 'diff_actual_and_model_pred', diff)
    new_df.insert(3, 'model_pred_ridership', model_pred)

    return new_df.sort_values(by='diff_actual_and_model_pred',
                                  ascending=True).head(30)
