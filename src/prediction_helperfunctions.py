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
    

    # check if part of string matches given output format to prompt -> Llama is annoying and won't only give the output format :(
    # pattern = r'\{[^{}]+\}'
    pattern = r'\{[^{}]*categorie[^{}]*\}'
    matches = re.findall(pattern, response)
    if len(matches) == 1:
        prediction_output = matches[0]
        predictions = [True if category.lower() in prediction_output.lower() else False for category in classes_list]

        # check if multiple classes were named, this is a prediction error
        if Counter(predictions)[True] > 1:
            return "MultiplePredictionErrorInOutput"

        # check if exactly one class is named, this is the prediction
        elif Counter(predictions)[True] == 1:
            prediction = [category.lower() for category in classes_list if category.lower() in prediction_output.lower()]
            return prediction[0]

        # if no class is named, then this is a no prediction error
        else:
            return 'NoPredictionInOutput'
        
    elif len(matches) > 1:
        return 'MultiplePredictionErrorInFormatting'
    else:
        return 'NoPredictionFormat'



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

"""
Given a dataframe with all docs that need to have prediction in the prediction_path file with the same run_id, 
return which rows of df have already been predicted and which have not. 
"""
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

""" Remove previous records of run_id and update with new scores. """
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
def check_data_split_input(subset_train, subset_test):
    if subset_train not in ['train', 'dev']:
        raise ValueError("subset_train must be either 'train' or 'dev'")
    if subset_test not in ['test', 'val']:
        raise ValueError("subset_test must be either 'test' or 'val'")
    
