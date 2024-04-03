from collections import Counter
import re
import time
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
        
""" Get the new runid """
def get_runid(path):

    # if not first run, set runid to most recent run+1
    if os.path.exists(path):
        df = pd.read_pickle(path)
        return max(df['run_id'])+1, df

    # if first run, set runid to 0
    else:
        return 0, pd.DataFrame()
    
""" Save evaluation metrics of a run """
def update_overview_results(df, model_name, subset=None):
    # df= dataframe with predictions for each do, one row per doc/prediction
    # model_name = string with the name of the model
    # subset = can be train, val, or test, or left open
 
    # get evalaution scores
    evaluation_dict = classification_report(df['label'], df['prediction'], output_dict=True)
    evaluation = pd.DataFrame(evaluation_dict).transpose()
    
    new_row = {
        # stuff about the run
        'run_id':df.iloc[0]['run_id'],
        'model':model_name,
        'prompt_function':df.iloc[0]['prompt_function'],
        'text_column':df.iloc[0]['text_column'],
        'date': get_datetime(),
        'runtime':sum(df['runtime']),
        'set':subset,
        'support':evaluation.iloc[-1]['support'],

        # evaluation
        'accuracy': evaluation_dict['accuracy'],

        'recall_weighted_avg':evaluation.loc[evaluation.index=='weighted avg']['recall'].values[0],
        'precision_weighted_avg': evaluation.loc[evaluation.index=='weighted avg']['precision'].values[0],
        'f1_weighted_avg': evaluation.loc[evaluation.index=='weighted avg']['f1-score'].values[0],

        'recall_macro_avg':evaluation.loc[evaluation.index=='macro avg']['recall'].values[0],
        'precision_macro_avg': evaluation.loc[evaluation.index=='macro avg']['precision'].values[0],
        'f1_macro_avg': evaluation.loc[evaluation.index=='macro avg']['f1-score'].values[0],


        'recall_classes': dict(zip(evaluation.index[0:-3], evaluation['recall'][0:-3])),
        'precision_classes': dict(zip(evaluation.index[0:-3], evaluation['precision'][0:-3])),
        'f1_classes': dict(zip(evaluation.index[0:-3], evaluation['f1-score'][0:-3])),
        'support_classes': dict(zip(evaluation.index[0:-3], evaluation['support'][0:-3])),

        # docs that were predicted
        'doc_paths':list(df['path'].values)
        
    }

    # create a new dataframe with the evaluation, each run has one row
    results = pd.DataFrame(columns=new_row.keys())
    results.loc[len(results)] = new_row
   
    # if not the first run, get results from previous runs
    path = f"{cf.output_path}/overview_results.pkl"
    if os.path.exists(path):
        earlier_results = pd.read_pickle(path)

        # combine evaluation of previous runs with current run
        results = pd.concat([earlier_results, results])

    # save to overview_results.pkl
    results.to_pickle(path)
   
