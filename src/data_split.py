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
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return final_df

# split the data into a balanced split
# for the original data this means -> test = 100 docs per class; train = the rest of the docs with max 1500 per class; train further split 90/10 into train and val. 
# for the demo data this means -> the same, except test = 1 docs per class. 
# does not actually save! 
def save_balanced_split(df, demo=False):
    if demo==False:
        n_test = 100
    else:
        n_test = 1

    # select randomly n_test docs for each class for the test set
    test_df = df.groupby('label').apply(lambda x: x.sample(n=n_test)).reset_index(drop=True)
    test_df['balanced_split'] = 'test'
    
    # select all doc ids that are not in the test set
    train_df = df.loc[~df['id'].isin(test_df['id'])]

    # select maximum of 1500 docs for each class for the training set
    train_df = train_df.groupby('label').apply(lambda x: x.sample(n=min(len(x), 1500))).reset_index(drop=True)

    # split train set further into train and validation
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])
    train_df['balanced_split'] = 'train'
    val_df['balanced_split'] = 'val'

    # combine all three sets
    balanced_df = pd.concat([test_df, train_df, val_df])
    remaining_df = df.loc[~df['id'].isin(balanced_df['id'])]
    remaining_df['balanced_split'] = 'discard'

    split_df = pd.concat([balanced_df, remaining_df])
    split_df = split_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return split_df
