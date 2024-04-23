""" Includes all the prompt templates """

# class list
def get_class_list():
    return ['Voordracht', 'Besluit', 'Schriftelijke Vragen', 'Brief', 'Raadsadres', 'Onderzoeksrapport', 'Termijnagenda', 'Raadsnotulen', 'Agenda', 'Motie', 'Actualiteit', 'Factsheets']

# ---- GEITje----------

def simple_prompt(doc,train_df, num_examples, text_column):
    prompt = f"""
    Classificeer het document in één van de categoriën.
    Houd het kort, geef enkel de naam van de categorie als response.
    
    Categoriën: {get_class_list()}
    
    Document: 
    {doc}
    
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