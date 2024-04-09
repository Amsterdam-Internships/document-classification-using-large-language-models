"""Paths when running on Azure (AML)"""
# Folder path from BlobFuse
blobfuse_path = '/home/azureuser/cloudfiles/code/blobfuse/'

# Output path to store stuff on Blobfuse
output_path = '/home/azureuser/cloudfiles/code/blobfuse/raadsinformatie/processed_data/woo_document_classification'

HUGGING_CACHE = "/home/azureuser/cloudfiles/code//hugging_cache"




####### REMOVE LATER ######
# Folder path containing PDF files
folder_path = '/home/azureuser/cloudfiles/code/blobfuse/raadsinformatie/raadsinformatie/search_results/voordrachten/'

# Folder containing folder with PDF files, these are the files annotated by Anna
folder_path_annotated = "/home/azureuser/cloudfiles/code/blobfuse/raadsinformatie/raadsinformatie/annotated/"

# CSV file containing the extracted texts from the files in folder_path_annotated
file_path_annotated_csv = '/home/azureuser/cloudfiles/code/Users/f.bakker/document-classification-using-large-language-models/notebooks/annotated.csv'

# Raadsinformatie folder, containing folder with all data
folder_path_raadsinformatie = '/home/azureuser/cloudfiles/code/blobfuse/raadsinformatie/raadsinformatie/'