""" Includes all the prompt templates """

# class list
def get_class_list():
    return ['Voordracht', 'Besluit', 'Schriftelijke Vragen', 'Brief', 'Raadsadres', 'Onderzoeksrapport', 'Termijnagenda', 'Raadsnotulen', 'Agenda', 'Motie', 'Actualiteit', 'Factsheets']

# ------ GEITje ------------
def simple_prompt(doc):
    prompt = f"""
    Classificeer het document in één van de categoriën.
    Houd het kort, geef enkel de naam van de categorie als response.
    
    Categoriën: {get_class_list()}
    
    Document: 
    {doc}
    
    """
    return prompt

def fewshot_prompt_examples(doc, extra_parameters):
    prompt = f"""
    Classificeer het document in één van de categoriën.
    Houd het kort, geef enkel de naam van de categorie als response.
    
    Categoriën: {get_class_list()}

    Voorbeeld document: {extra_parameters['example']}
    Label: {extra_parameters['label']}
    
    Document: 
    {doc}
    
    """
    return prompt