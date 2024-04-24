from sklearn.model_selection import train_test_split
import pandas as pd

"""
Function takes a dataframe and splits the data into train, test, val and dev set.
Only need to run it once.
"""
def save_split(df):
    train_df, temp_df = train_test_split(df, test_size=0.25, random_state=42)

    # Splitting temp into test (20%) and val_dev (5%)
    test_df, val_dev_df = train_test_split(temp_df, test_size=0.2, random_state=42)

    # Splitting val_dev into validation (1%) and development (4%)
    dev_df,val_df = train_test_split(val_dev_df, test_size=0.2, random_state=42)

    # set split into 4 ways: train, test, val and dev
    train_df['4split'] = 'train'
    test_df['4split'] = 'test'
    val_df['4split'] = 'val'
    dev_df['4split'] = 'dev'

    # set split into 2 ways: test and training
    train_df['2split'] = 'train'
    test_df['2split'] = 'test'
    val_df['2split'] = 'test'
    dev_df['2split'] = 'train'

    # Combining the DataFrames
    final_df = pd.concat([train_df, test_df, val_df, dev_df])
    return final_df