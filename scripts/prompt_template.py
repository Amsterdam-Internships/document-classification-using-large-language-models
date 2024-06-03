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


# def fewshot_prompt_bm25(doc, train_df, num_examples, text_column, BM25_model):
#     # select all texts in train_df, these are possible examples
#     examples =list(train_df[text_column].values)

#     # calculate BM25 scores for each example
#     scores = np.argsort(BM25_model.transform(doc, [item for item in examples]))[::-1]

#     # select top examples
#     bm25_examples = [examples[score] for score in scores[:num_examples]]

#     # start prompt with instructions
#     instruction = ("Classificeer het document in één van de categoriën. "+
#     f"Eerst krijg je een lijst met mogelijke categoriën, daarna {num_examples} voorbeelden van documenten en tot slot het document dat gecategoriseerd moet worden. " +
#     f"Categoriën: {get_class_list()}. "
#     )

#     # include examples in prompt
#     for ex in range(len(bm25_examples)):
#         example = bm25_examples[ex]
#         label = train_df.loc[train_df[text_column]==example].iloc[0]['label']
#         mini_prompt =(
#             f"Voorbeeld document {ex+1}: " + 
#             f"{example} \n" +
#             f"Output van voorbeeld document {ex+1}: {{'categorie': {label}}} \n")
        
#         instruction += mini_prompt

#     doc_prompt = get_doc_prompt(doc)
#     # prompt = SYS_MES_CONTEXT + instruction + doc_prompt + "[/INST]"
#     prompt = "<s>[INST] " + instruction + doc_prompt + "[/INST]"
#     return prompt


# fewshot prompt
# def fewshot_prompt_bm25(doc, train_df, num_examples, text_column, BM25_model):
#     # select all texts in train_df, these are possible examples
#     examples =list(train_df[text_column].values)

#     # calculate BM25 scores for each example
#     scores = np.argsort(BM25_model.transform(doc, [item for item in examples]))[::-1]

#     # select top examples
#     bm25_examples = [examples[score] for score in scores[:num_examples]]

#     # start prompt with instructions
#     instruction = ("Het is jouw taak om een document te classificeren in één van de categoriën. "+
#     f"Eerst krijg je een lijst met mogelijke categoriën, daarna {num_examples} voorbeelden van documenten en tot slot het document dat gecategoriseerd moet worden. " +
#     f"Categoriën: {get_class_list()}. "
#     )

#     # include examples in prompt
#     for ex in range(len(bm25_examples)):
#         example = bm25_examples[ex]
#         label = train_df.loc[train_df[text_column]==example].iloc[0]['label']
#         mini_prompt =(
#             f"Voorbeeld document {ex+1}: " + 
#             f"{example} \n" +
#             f"Output van voorbeeld document {ex+1}: {{'categorie': {label}}} \n")
        
#         instruction += mini_prompt

#     doc_prompt = get_doc_prompt(doc)
#     prompt = SYS_MES_CONTEXT + instruction + doc_prompt + "[/INST]"
#     return prompt

### OLD code


# SYS_MES_CONTEXT = ("<s>[INST] <<SYS>> Hieronder staat een instructie die een taak beschrijft, " +
#         "gekoppeld aan input die verdere context biedt. "+
#         "Schrijf een reactie die de taak op passende wijze voltooit.<</SYS>> ")

# SYS_MES = ("<s>[INST] <<SYS>> Hieronder staat een instructie die een taak beschrijft. " +
#         "Schrijf een reactie die de taak op passende wijze voltooit.<</SYS>> ")



# def geitje_simple_prompt(doc):
#     instruction = (f"<|user|>\n"+ 
#     f"Classificeer het document in één van de categoriën. " +
#     f"Categoriën: {get_class_list()}. ")
    
#     doc_prompt = get_doc_prompt(doc)
#     prompt = instruction + doc_prompt + '</s>'
#     return prompt

# def simple_prompt(doc, remove_template=False):
#     instruction = (f"Classificeer het document in één van de categoriën. " +
#     f"Categoriën: {get_class_list()}. ")

#     doc_prompt = get_doc_prompt(doc)

#     if remove_template==False:
#         prompt = SYS_MES + instruction + doc_prompt + "[/INST]"
#     elif remove_template==True:
#         prompt = instruction + doc_prompt

#     return prompt

# def simple_prompt_v2(doc):
#     instruction = (f"<s>[INST] Classificeer het document in één van de categoriën. " +
#     f"Categoriën: {get_class_list()}. ")
#     doc_prompt = get_doc_prompt(doc)
#     prompt = instruction + doc_prompt+"[/INST]"
#     return prompt

# def OldSimple_prompt(doc):
#     prompt = f"""
#     Classificeer het document in één van de categoriën.
#     Geef de output in de vorm van een JSON file: {{'categorie': categorie van het document}}
    
#     Categoriën: {get_class_list()}
    
#     Document: 
#     {doc}

#     Vul in met de categorie van het document: {{'categorie': ??}}     
#     """
#     return prompt



# def geitje_fewshot_prompt(doc, train_df, num_examples, text_column, BM25_model):
#     # select all texts in train_df, these are possible examples
#     examples =list(train_df[text_column].values)

#     # calculate BM25 scores for each example
#     scores = np.argsort(BM25_model.transform(doc, [item for item in examples]))[::-1]

#     # select top examples
#     bm25_examples = [examples[score] for score in scores[:num_examples]]

#     # start prompt with instructions
#     instruction = ("<|user|>\n"+ "Het is jouw taak om een document te categoriseren in één van de categoriën. "+
#     f"Eerst krijg je een lijst met mogelijke categoriën, daarna {num_examples} voorbeelden van documenten en tot slot het document dat gecategoriseerd moet worden. " +
#     f"Categoriën: {get_class_list()}. "
#     )

#     # include examples in prompt
#     for ex in range(len(bm25_examples)):
#         example = bm25_examples[ex]
#         label = train_df.loc[train_df[text_column]==example].iloc[0]['label']
#         mini_prompt =(
#             f"Voorbeeld document {ex+1}: " + 
#             f"{example} \n" +
#             f"Output van voorbeeld document {ex+1}: {{'categorie': {label}}} \n")
        
#         instruction += mini_prompt

#     doc_prompt = get_doc_prompt(doc)
#     prompt = instruction + doc_prompt + "</s>"
#     return prompt


# def mistral_llama_fewshot_prompt(doc, train_df, num_examples, text_column, BM25_model):
#     # select all texts in train_df, these are possible examples
#     examples =list(train_df[text_column].values)

#     # calculate BM25 scores for each example
#     scores = np.argsort(BM25_model.transform(doc, [item for item in examples]))[::-1]

#     # select top examples
#     bm25_examples = [examples[score] for score in scores[:num_examples]]

#     # start prompt with instructions
#     instruction = ("<s>[INST] Het is jouw taak om een document te categoriseren in één van de categoriën. "+
#     f"Eerst krijg je een lijst met mogelijke categoriën, daarna {num_examples} voorbeelden van documenten en tot slot het document dat gecategoriseerd moet worden. " +
#     f"Categoriën: {get_class_list()}. "
#     )

#     # include examples in prompt
#     for ex in range(len(bm25_examples)):
#         example = bm25_examples[ex]
#         label = train_df.loc[train_df[text_column]==example].iloc[0]['label']
#         mini_prompt =(
#             f"Voorbeeld document {ex+1}: " + 
#             f"{example} \n" +
#             f"Output van voorbeeld document {ex+1}: {{'categorie': {label}}} \n")
        
#         instruction += mini_prompt

#     doc_prompt = get_doc_prompt(doc)
#     prompt = instruction + doc_prompt + "[/INST]"
#     return prompt


# def OldFewshot_prompt_bm25(doc, train_df, num_examples, text_column, BM25_model):
#     # select all texts in train_df, these are possible examples
#     examples =list(train_df[text_column].values)

#     # calculate BM25 scores for each example
#     scores = np.argsort(BM25_model.transform(doc, [item for item in examples]))[::-1]

#     # select top examples
#     bm25_examples = [examples[score] for score in scores[:num_examples]]

#     # start prompt with instructions
#     prompt = f"""
#     Classificeer het document in één van de categoriën.
#     Geef de output in de vorm van een JSON file: {{'categorie': categorie van het document}}
#     Eerst krijg je een lijst met mogelijke categoriën, daarna {num_examples} voorbeelden van documenten en tot slot het document dat gecategoriseerd moet worden. 

#     Categoriën: {get_class_list()}
#     """

#     # include examples in prompt
#     for ex in range(len(bm25_examples)):
#         example = bm25_examples[ex]
#         label = train_df.loc[train_df[text_column]==example].iloc[0]['label']
#         mini_prompt = f"""
#         \n
#         Voorbeeld document {ex+1}:
#         "{example}"
#         \n
#         Output van voorbeeld document {ex+1}: {{'categorie': {label}}}  
#         """
#         prompt += mini_prompt

#     # give doc to classify
#     doc_prompt = f"""
#     Categoriseer dit document:
#         {doc}

#     Vul in met de categorie van het document: {{'categorie': ??}}    
#     """

#     prompt += doc_prompt
#     return prompt

# ## OLD prompts

# def simple_prompt(doc):
#     prompt = f"""
#     Classificeer het document in één van de categoriën.
#     Geef de output in de vorm van een JSON file: {{'categorie': categorie van het document}}
    
#     Categoriën: {get_class_list()}
    
#     Document: 
#     {doc}

#     Vul in met de categorie van het document: {{'categorie': ??}}     
#     """
#     return prompt

# def fewshot_prompt_bm25(doc, train_df, num_examples, text_column, BM25_model):
#     # select all texts in train_df, these are possible examples
#     examples =list(train_df[text_column].values)

#     # calculate BM25 scores for each example
#     scores = np.argsort(BM25_model.transform(doc, [item for item in examples]))[::-1]

#     # select top examples
#     bm25_examples = [examples[score] for score in scores[:num_examples]]

#     # start prompt with instructions
#     prompt = f"""
#     Het is jouw taak om een document te categoriseren in één van de categoriën.
#     Eerst krijg je een lijst met mogelijke categoriën, daarna {num_examples} voorbeelden van documenten en tot slot het document dat gecategoriseerd moet worden. 

#     Categoriën: {get_class_list()}
#     """

#     # include examples in prompt
#     for ex in range(len(bm25_examples)):
#         example = bm25_examples[ex]
#         label = train_df.loc[train_df[text_column]==example].iloc[0]['label']
#         mini_prompt = f"""
#         \n
#         Voorbeeld document {ex+1}:
#         "{example}"
#         \n
#         Output van voorbeeld document {ex+1}: {{'categorie': {label}}}  
#         """
#         prompt += mini_prompt

#     # give doc to classify
#     doc_prompt = f"""
#     Categoriseer dit document:
#         {doc}

#     Geef de output in de vorm van een JSON file: {{'categorie': categorie van het document}}
#     Vul in met de categorie van het document: {{'categorie': ??}}    
#     """

#     prompt += doc_prompt
#     return prompt