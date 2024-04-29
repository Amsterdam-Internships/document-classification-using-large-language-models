import numpy as np


""" Includes all the prompt templates """
# class list
def get_class_list():
    return ['Voordracht', 'Besluit', 'Schriftelijke Vraag', 'Brief', 'Raadsadres', 'Onderzoeksrapport', 'Termijnagenda', 'Raadsnotulen', 'Agenda', 'Motie', 'Actualiteit', 'Factsheet']

# ---- GEITje----------
# simple_prompt takes extra input, since those parameters are needed for fewshot prompt. Allows to run same code for experiment. 
def simple_prompt(doc):
    prompt = f"""
    Classificeer het document in één van de categoriën.
    Geef de output in de vorm van een JSON file: {{'categorie': categorie van het document}}
    
    Categoriën: {get_class_list()}
    
    Document: 
    {doc}

    Vul in met de categorie van het document: {{'categorie': ??}}     
    """
    return prompt


def fewshot_prompt_examples(doc, train_df, num_examples, text_column):
    examples = train_df.sample(n=num_examples)

    prompt = f"""
    Het is jouw taak om een document te categoriseren in één van de categoriën.
    Eerst krijg je een lijst met mogelijke categoriën, daarna {num_examples} voorbeelden van documenten en tot slot het document dat gecategoriseerd moet worden. 
    
    Categoriën: {get_class_list()}
    """

    for index, row in examples.iterrows():
        mini_prompt = f"""
    Dit is een voorbeeld document de categorie {row['label']}:
        {row[text_column]}
        """

        prompt += mini_prompt

    doc_prompt = f"""
    Categoriseer dit document:
        {doc}
    """

    prompt += doc_prompt
    return prompt


def fewshot_prompt_bm25(doc, train_df, num_examples, text_column, BM25_model):
    # select all texts in train_df, these are possible examples
    examples =list(train_df[text_column].values)

    # calculate BM25 scores for each example
    scores = np.argsort(BM25_model.transform(doc, [item for item in examples]))[::-1]

    # select top examples
    bm25_examples = [examples[score] for score in scores[:num_examples]]

    # start prompt with instructions
    prompt = f"""
    Het is jouw taak om een document te categoriseren in één van de categoriën.
    Eerst krijg je een lijst met mogelijke categoriën, daarna {num_examples} voorbeelden van documenten en tot slot het document dat gecategoriseerd moet worden. 

    Categoriën: {get_class_list()}
    """

    # include examples in prompt
    for ex in range(len(bm25_examples)):
        example = bm25_examples[ex]
        label = train_df.loc[train_df[text_column]==example].iloc[0]['label']
        mini_prompt = f"""
        \n
        Voorbeeld document {ex+1}:
        "{example}"
        \n
        Output van voorbeeld document {ex+1}: {{'categorie': {label}}}  
        """
        prompt += mini_prompt

    # give doc to classify
    doc_prompt = f"""
    Categoriseer dit document:
        {doc}

    Geef de output in de vorm van een JSON file: {{'categorie': categorie van het document}}
    Vul in met de categorie van het document: {{'categorie': ??}}    


    """

    prompt += doc_prompt
    return prompt

