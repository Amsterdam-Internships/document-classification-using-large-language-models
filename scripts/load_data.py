

## DONT NEED THIS FILE??

# import pandas as pd
# import nltk
# from nltk.tokenize import word_tokenize


# """
# Function takes a df with path to txt files, and load the text from the txt files.
# It's possible to only load part of the data.
# Load -> a list containing which part of the data to load, ['train', 'test', 'val', 'complete']
# """

# def load_txt_files(input_df, load):
#     # load is a list containing which part of the data to load, ['train', 'test', 'val', 'complete']
#     df = input_df.copy()

#     # create empty dataframe
#     columns_list = list(df.columns.values)
#     columns_list.extend(['text', 'tokens', 'token_count'])
#     return_df = pd.DataFrame(columns=columns_list)

#     # if load is complete dataset, load all txt files
#     if load==['complete']:
#         subdf = df.copy()

#      # if load is subsections of the data - examples: ['train', 'test'], ['train'], ['test'], ['train', 'val']
#     else: 
#         subdf = df.loc[df['set'].isin(load)]

#     for index, row in subdf.iterrows():
#         # extract text
#         with open(row['path']) as txt_file:
#             text = txt_file.read()

#         # check if text is longer than 5 characters
#         if len(text) > 5:
#             tokens = word_tokenize(text)
#             len_tokens = len(tokens)

#             # save in dataframe
#             return_df.loc[len(return_df)] = {'label':row['label'], 'path':row['path'], 'id':row['id'],'set':row['set'],'text': text, 'tokens':tokens, 'token_count':len_tokens}

#     return return_df

# # def load_txt_files(input_df, load):
# #     df = input_df.copy()

# #     # create empty dataframe
# #     columns_list = list(df.columns.values)
# #     columns_list.append('text')
# #     return_df = pd.DataFrame(columns=columns_list)

# #     # if load is complete dataset, load all txt files
# #     if load==['complete']:
# #         subdf = df.copy()

# #      # if load is subsections of the data - examples: ['train', 'test'], ['train'], ['test'], ['train', 'val']
# #     else: 
# #         subdf = df.loc[df['set'].isin(load)]

# #     for index, row in subdf.iterrows():
# #         # extract text
# #         with open(row['path']) as txt_file:
# #             text = txt_file.read()

# #         # save in dataframe
# #         return_df.loc[len(return_df)] = {'label':row['label'], 'path':row['path'], 'id':row['id'],'set':row['set'],'text': text}

# #     return return_df
 

# # txt_files_df = load_txt_files(txtfile_paths,  ['train'])

# """
# How to load the data:

#     import sys
#     sys.path.append('../scripts/') 
#     from load_data import load_txt_files
    
# """