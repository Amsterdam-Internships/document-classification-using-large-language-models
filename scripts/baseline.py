
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import prediction_helperfunctions as ph
import time
import pandas as pd

"""
This file contains functions to run the baselines.
"""



"""Function returns X and y set for either the train, val, test or dev set."""
def load_data_split(df, split_col,subset, label_col):
    subdf = df.loc[df[split_col]==subset]
    X = subdf.drop(columns=[label_col])
    y = subdf[label_col]
    return X, y

"""Function saves predictions and results of a baseline in two seperate files."""
def run_baseline(baseline_function,model_name, dataframe,split_col, subset_train, subset_test, text_col, label_col, prediction_path, overview_path):
    start_time = time.time()

    ph.check_data_split_input(subset_train, subset_test)
    X_train, y_train = load_data_split(dataframe,split_col,subset_train,label_col) 
    X_test, y_test = load_data_split(dataframe,split_col,subset_test,label_col) 

    # use TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X_train_tfidf_bin = vectorizer.fit_transform(X_train[text_col])
    X_test_tfidf_bin = vectorizer.transform(X_test[text_col])

    # train classifier on training data
    model = baseline_function
    model.fit(X_train_tfidf_bin, y_train)

    # get predictions
    y_pred = model.predict(X_test_tfidf_bin)

    # get classification report
    report = classification_report(y_test, y_pred)
    print(report)

    # save data about predictions
    date = ph.get_datetime()
    predictions = X_test.copy()
    predictions[label_col] = y_test
    predictions['prediction'] = y_pred
    predictions['model'] = model_name
    predictions['date'] = date

    # remove unneccary columns
    columns =['set', 'text', 'tokens', 'token_count', 'clean_tokens', 'clean_tokens_count', 'pdf_path', 'clean_text', 'token_count_geitje', 'token_count_mistral', 'token_count_llama2_7b_hf', '4split', '2split']
    remove_col = [col for col in columns if col in dataframe.columns]
    predictions = predictions.drop(columns=remove_col)

    # save predictions
    ph.combine_and_save_df(predictions, prediction_path)

    # save run -> scores + runtime
    overview = pd.DataFrame(
        [{
            'model':model_name,
            'date': date,
            'train_set': subset_train,
            'test_set': subset_test,
            'train_set_support':len(X_train),
            'test_set_support':len(X_test),
            'split_col':split_col,
            'text_col':X_train.iloc[0]['trunc_col'],
            'runtime':time.time()-start_time,
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_avg_precision': precision_score(y_test, y_pred, average='macro'),
            'macro_avg_recall': recall_score(y_test, y_pred, average='macro'),
            'macro_avg_f1': f1_score(y_test, y_pred, average='macro'),
            'classification_report':report
        }   ]
    )
    ph.combine_and_save_df(overview, overview_path)

    return predictions


