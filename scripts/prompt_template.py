""" Includes all the prompt templates """

# class list
def get_class_list():
    return ['Voordracht', 'Besluit', 'Schriftelijke Vragen', 'Brief', 'Raadsadres', 'Onderzoeksrapport', 'Termijnagenda', 'Raadsnotulen', 'Agenda', 'Motie', 'Actualiteit', 'Factsheets']

# ------ GEITje ------------
def simple_prompt(classes, doc):
    prompt = f"""
    Classificeer het document in één van de categoriën.
    Houd het kort, geef enkel de naam van de categorie als response.
    
    Categoriën: {classes}
    
    Document: 
    {doc}
    
    """
    return prompt