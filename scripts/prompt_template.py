""" Includes all the prompt templates """

# ------ GEITje ------------
def simple_prompt(classes, doc):
    prompt = f"""
    Classificeer het document in één van de categoriën.
    
    Categoriën: {classes}
    
    Document: 
    {doc}
    
    """
    return prompt