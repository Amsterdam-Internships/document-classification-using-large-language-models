import numpy as np


""" Includes all the prompt templates """
# class list
def get_class_list():
    return ['Voordracht', 'Besluit', 'Schriftelijke Vraag', 'Brief', 'Raadsadres', 'Onderzoeksrapport', 'Raadsnotulen', 'Agenda', 'Motie', 'Actualiteit', 'Factsheet']

# funcation to format document
def get_doc_prompt(doc):
    doc_prompt = (f"Categoriseer dit document: "+
                f"{doc}\n" +
                f"Geef de output in de vorm van een JSON file: {{'categorie': categorie van het document}}. "
                )
    return doc_prompt

# zeroshot prompt used for mistral and llama. Also used for data formatting for FT. 
# original name: simple_prompt_v2
def zeroshot_prompt_mistral_llama(doc, remove_template=True):
    instruction = (f"Classificeer het document in één van de categoriën. " +
    f"Categoriën: {get_class_list()}. ")

    doc_prompt = get_doc_prompt(doc)

    if remove_template==False:
        prompt = "<s>[INST] " + instruction + doc_prompt + "[/INST]"
    elif remove_template==True:
        prompt = instruction + doc_prompt

    return prompt


# zeroshot prompt used for GEITje
# original name: OldSimple_prompt
def zeroshot_prompt_geitje(doc):
    prompt = f"""
    Classificeer het document in één van de categoriën.
    Geef de output in de vorm van een JSON file: {{'categorie': categorie van het document}}
    
    Categoriën: {get_class_list()}
    
    Document: 
    {doc}

    Vul in met de categorie van het document: {{'categorie': ??}}     
    """
    return prompt


def fewshot_prompt_with_template(doc, train_df, num_examples, text_column, BM25_model):
    # select all texts in train_df, these are possible examples
    examples =list(train_df[text_column].values)

    # calculate BM25 scores for each example
    scores = np.argsort(BM25_model.transform(doc, [item for item in examples]))[::-1]

    # select top examples
    bm25_examples = [examples[score] for score in scores[:num_examples]]

    # start prompt with instructions
    instruction = ("Classificeer het document in één van de categoriën. "+
    f"Eerst krijg je een lijst met mogelijke categoriën, daarna {num_examples} voorbeelden van documenten en tot slot het document dat gecategoriseerd moet worden. " +
    f"Categoriën: {get_class_list()}. "
    )

    # include examples in prompt
    for ex in range(len(bm25_examples)):
        example = bm25_examples[ex]
        label = train_df.loc[train_df[text_column]==example].iloc[0]['label']
        mini_prompt =(
            f"Voorbeeld document {ex+1}: " + 
            f"{example} \n" +
            f"Output van voorbeeld document {ex+1}: {{'categorie': {label}}} \n")
        
        instruction += mini_prompt

    doc_prompt = get_doc_prompt(doc)
    prompt = "<s>[INST] " + instruction + doc_prompt + "[/INST]"
    return prompt


def fewshot_prompt_no_template(doc, train_df, num_examples, text_column, BM25_model):
    # select all texts in train_df, these are possible examples
    examples =list(train_df[text_column].values)

    # calculate BM25 scores for each example
    scores = np.argsort(BM25_model.transform(doc, [item for item in examples]))[::-1]

    # select top examples
    bm25_examples = [examples[score] for score in scores[:num_examples]]

    # start prompt with instructions
    instruction = ("Classificeer het document in één van de categoriën. "+
    f"Eerst krijg je een lijst met mogelijke categoriën, daarna {num_examples} voorbeelden van documenten en tot slot het document dat gecategoriseerd moet worden. " +
    f"Categoriën: {get_class_list()}. "
    )

    # include examples in prompt
    for ex in range(len(bm25_examples)):
        example = bm25_examples[ex]
        label = train_df.loc[train_df[text_column]==example].iloc[0]['label']
        mini_prompt =(
            f"Voorbeeld document {ex+1}: " + 
            f"{example} \n" +
            f"Output van voorbeeld document {ex+1}: {{'categorie': {label}}} \n")
        
        instruction += mini_prompt

    doc_prompt = get_doc_prompt(doc)
    prompt = instruction + doc_prompt
    return prompt
