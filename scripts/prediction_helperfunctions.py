from collections import Counter
import re
import datetime
import pytz
import os
from sklearn.metrics import classification_report
import prompt_template as pt
import pandas as pd
import config_azure as cf

"""
This file contains functions that are used to make predictions and save them. 
These functions can be used for all models/prompts. 
"""


""" Given the string response, extract the prediction """
def get_prediction_from_response(response):
    # get a list of the possible classes
    classes_list = pt.get_class_list()

    predictions = [True if category.lower() in response.lower() else False for category in classes_list]

    # check if multiple classes were named, this is a prediction error
    if Counter(predictions)[True] > 1:
        return "PredictionError"

    # check if exactly one class is named, this is the prediction
    elif Counter(predictions)[True] == 1:
        prediction = [category.lower() for category in classes_list if category.lower() in response.lower()]
        return prediction[0]

    # if no class is named, then this is a no prediction error
    else:
        return 'NoPrediction'



""" Extract the promptfunction name """
def get_promptfunction_name(prompt_function):
    string = f"{prompt_function}"
    match = re.search(r'<function\s+(\w+)', string)
    if match:
        function_name = match.group(1)
        return function_name
    else:
        return f"{prompt_function}"
    


""" Get the current time in the Netherlands """
def get_datetime():
    current_datetime_utc = datetime.datetime.now(pytz.utc)

    # Convert UTC time to Dutch time (CET)
    dutch_timezone = pytz.timezone('Europe/Amsterdam')
    current_datetime_dutch = current_datetime_utc.astimezone(dutch_timezone)
    return current_datetime_dutch
        

""" Combine new dataframe with dataframe already saved, if file already exists. """
def combine_and_save_df(model_df, save_to_path):
    
    # combine with earlier runs if exists
    if os.path.exists(save_to_path):
        original = pd.read_pickle(save_to_path)
        model_df = pd.concat([original, model_df])

    model_df.to_pickle(save_to_path)


def get_rows_to_predict(df, prediction_path, run_id):
    if os.path.exists(prediction_path):
        previous_predictions = pd.read_pickle(prediction_path)
        previous_predictions = previous_predictions.loc[previous_predictions['run_id']==run_id]
        to_predict = df.loc[~df['path'].isin(previous_predictions['path'])]

        if len(previous_predictions) != 0:
            print("Run-id already known, resuming predictions...")

        elif len(to_predict) == 0:
            print("ALL PREDICTIONS HAVE BEEN MADE")

    else:
        to_predict = df.copy()
        previous_predictions = 'None'

    return to_predict, previous_predictions

def replace_and_save_df(overview, overview_path, run_id):
    if os.path.exists(overview_path):
        previous_runs = pd.read_pickle(overview_path)

        # remove previous results with same run_id and replace with new results
        previous_runs = previous_runs.loc[previous_runs['run_id']!= run_id]
        update = pd.concat([previous_runs, overview])
        update.to_pickle(overview_path)

    else:
        overview.to_pickle(overview_path)
   

""" Raise error if input is incorrect """
def check_input_set_run_id(set_run_id):
    if set_run_id not in ['new', 'previous'] and not isinstance(set_run_id, int):
        raise ValueError("set_run_id must be either 'new' or 'previous' or an integer.")

def check_input_subset(subset):
    if subset not in [None, 'train', 'val', 'test', 'complete']:
            raise ValueError("set_run_id must be one of these options [None, 'train', 'val', 'test', 'complete']")
    
def check_data_split_input(subset_train, subset_test):
    if subset_train not in ['train', 'dev']:
        raise ValueError("subset_train must be either 'train' or 'dev'")
    if subset_test not in ['test', 'val']:
        raise ValueError("subset_test must be either 'test' or 'val'")
    




# """ Get the new runid """
# def get_runid(path):

#     # if not first run, set runid to most recent run+1
#     if os.path.exists(path):
#         df = pd.read_pickle(path)
#         return max(df['run_id'])+1, df

#     # if first run, set runid to 0
#     else:
#         return 0, pd.DataFrame()

# """ Get the new runid """
# def get_runid(set_run_id):
#     path = f"{cf.output_path}/overview_results.pkl"
    
#     # if not run_id given as in, then select new, or previous run_id
#     if not isinstance(set_run_id, int):

#         # if not first run, set runid to most recent run+1
#         if os.path.exists(path):
#             df = pd.read_pickle(path)

#             if set_run_id == 'new':
#                 return max(df['run_id'])+1

#             # if want resume run, return current run_id
#             else:
#                 return max(df['run_id'])

#         # if first run, set runid to 0
#         else:
#             return 0
        
#     # if run_id is given. i.e. its a integer
#     else:
#         return set_run_id
    
# """ Combine the current predictions with previous runs """
# def combine_current_with_previous_predictions(path_to_old_predictions, new_predictions):

#     # if predictions are already made, load and combine
#     if os.path.exists(path_to_old_predictions):
#         predictions_df = pd.read_pickle(path_to_old_predictions)
#         all_predictions = pd.concat([predictions_df, new_predictions])

#     # no previous predictions, return the new predictions
#     else:
#         all_predictions = new_predictions
    
#     return all_predictions


""" Save evaluation metrics of a run """
# def update_overview_results(df, model_name,save_predictions_path, subset=None):
#     # df= dataframe with predictions for each do, one row per doc/prediction
#     # model_name = string with the name of the model
#     # subset = can be train, val, or test, or left open
 
#     # get evalaution scores
#     evaluation_dict = classification_report(df['label'], df['prediction'], output_dict=True)
#     evaluation = pd.DataFrame(evaluation_dict).transpose()
    
#     current_rundid = df.iloc[0]['run_id']

#     new_row = {
#         # stuff about the run
#         'run_id':current_rundid,
#         'model':model_name,
#         'prompt_function':df.iloc[0]['prompt_function'],
#         'text_column':df.iloc[0]['text_column'],
#         'saved_in_path':save_predictions_path,
#         'date': get_datetime(),
#         'runtime':sum(df['runtime']),
#         'set':subset,
#         'support':evaluation.iloc[-1]['support'],

#         # evaluation
#         'accuracy': evaluation_dict['accuracy'],

#         # 'recall_weighted_avg':evaluation.loc[evaluation.index=='weighted avg']['recall'].values[0],
#         # 'precision_weighted_avg': evaluation.loc[evaluation.index=='weighted avg']['precision'].values[0],
#         # 'f1_weighted_avg': evaluation.loc[evaluation.index=='weighted avg']['f1-score'].values[0],

#         'recall_macro_avg':evaluation.loc[evaluation.index=='macro avg']['recall'].values[0],
#         'precision_macro_avg': evaluation.loc[evaluation.index=='macro avg']['precision'].values[0],
#         'f1_macro_avg': evaluation.loc[evaluation.index=='macro avg']['f1-score'].values[0],


#         # 'recall_classes': dict(zip(evaluation.index[0:-3], evaluation['recall'][0:-3])),
#         # 'precision_classes': dict(zip(evaluation.index[0:-3], evaluation['precision'][0:-3])),
#         # 'f1_classes': dict(zip(evaluation.index[0:-3], evaluation['f1-score'][0:-3])),
#         # 'support_classes': dict(zip(evaluation.index[0:-3], evaluation['support'][0:-3])),

#         # docs that were predicted
#         'doc_paths':list(df['path'].values)
        
#     }

#     # create a new dataframe with the evaluation, each run has one row
#     results = pd.DataFrame(columns=new_row.keys())
#     results.loc[len(results)] = new_row
   
#     # if not the first run, get results from previous runs
#     path = f"{cf.output_path}/overview_results.pkl"
#     if os.path.exists(path):
#         earlier_results = pd.read_pickle(path)

#         # remove records, if run_id is resumed.
#         earlier_results = earlier_results.loc[earlier_results['run_id']!= current_rundid]

#         # combine evaluation of previous runs with current run
#         results = pd.concat([earlier_results, results])

#     display(results.iloc[-1])

#     # save to overview_results.pkl
#     results.to_pickle(path)


    

