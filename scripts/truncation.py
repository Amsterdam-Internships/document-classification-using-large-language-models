import pandas as pd
 
"""
Function to add a truncated version of the text. Possible to give different token thresholds, both for the front of the document and the back of the document. 
"""
def add_truncation_column(df,text_col, token_col_name, front_token_threshold, back_token_threshold=0):
    input = []
    for index, row in df.iterrows():
        # select text according to the token threshold -> first FRONT
        # select first n (= token_theshold) tokens using the model tokenizer
        tokens = row[token_col_name][0:front_token_threshold]

        # combine tokens into txt
        tokens_txt = ''.join(tokens)

        # \n is converted by tokenizer to <0x0A>, we reverse this to get original length
        len_char = len(tokens_txt.replace("<0x0A>", "\n")) # get character length

        # select the same amount of characters as the tokens
        front_txt = row[text_col][0:len_char]

        # Check if back of document also given as input
        if back_token_threshold != 0:
            # select LAST n (= token_theshold) tokens using the model tokenizer
            tokens = row[token_col_name][-back_token_threshold:]

            # combine tokens into txt
            tokens_txt = ''.join(tokens)

            # \n is converted by tokenizer to <0x0A>, we reverse this to get original length
            len_char = len(tokens_txt.replace("<0x0A>", "\n")) # get character length

            # select the same amount of characters as the tokens
            back_txt = row[text_col][-len_char:]

            # combine front and back text
            input_txt = front_txt + ' ' + back_txt

        else:
            input_txt = front_txt

        input.append(input_txt)

    df['trunc_txt'] = input
    df['trunc_col'] = f"Truncation{token_col_name}Front{front_token_threshold}Back{back_token_threshold}"
    return df




# trunc = add_truncation_column(tok, 'text', 'GEITjeTokens', 50,50)
# display(trunc)