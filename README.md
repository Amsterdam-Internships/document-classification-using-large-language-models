# Document Classification using Large Language Models

This repo is for code related to the project Document Classification under the Woo (Open Government Act). The goal of this research is to find an effective method to classify documents that the Municipality of Amsterdam publishes. The focus will be on Dutch Large Language Models.

## Overleaf Link
[Thesis Design](https://www.overleaf.com/2549441224szvvffnxqsdk#eda3e6)


[Thesis](https://www.overleaf.com/8368827141bwgxbjwcfgfv#3d9efc)

## Background



## Folder Structure

* [`data`](./data): Sample data for demo purposes
* [`docs`](./docs): If main [README.md](./README.md) is not enough
* [`notebooks`](./notebooks): Jupyter notebooks / tutorials
* [`res`](./res): Relevant resources, e.g. [`images`](./res/images/) for the documentation
* [`scripts`](./scripts): Scripts for automating tasks
* [`src`](./src): All sourcecode files specific to this project
* [`tests`](./tests) Unit tests
* ...






## Installation 

1) Clone this repository:



```bash
git clone https://github.com/AmsterdamInternships/document-classification-using-large-language-models.git
```




2) Install all dependencies:
    
Right now, im having trouble with setting up the requirements.txt files. I will get to that later, for now, the most important libraries to install to get the fine-tuning to work are:

```bash
- pip install torch
- pip install datasets
- pip install transformers
- pip install trl
- pip install accelerate 
- pip install sentencepiece
- pip install jupyter
- pip install protobuf 
pip install bitsandbytes
pip install bnb
pip install wandb==0.13.3 --upgrade
pip install tensorboardX
```


```bash
pip install -r requirements.txt
```



The code has been tested with Python 3.9 on Windows. 

## Usage

## How it works

Run the following order of the notebooks:
- load_txt -> loads the data from the ocr files and splits the data into subsets (train, test, dev, val)
- clean_data -> check shortest docs and remove messy data.
- baseline  -> run baselines on complete docs.
- text truncation -> truncate text using tokenizer of either Llama or Mistral (Geitje has the same as mistral) and try out thresholds on baselines.
- dataFormattingFinetuning  -> format data into DatasetDict, and push to huggingface
- CLgeitje -> notebook to run GEITje In-Context learning
- FTgeitje. still very messy, work in progress. Ill clean up after fixing the bug, will probably make more of mess anyway. 

Stand alone notebooks:
-EDA/EDA_clean_for_submission
- data_insight -> still messy. use it now to get some quick results for either overleaf or to check something in the data.


Can be divided in subsections:

### input
### algorithm
### output

OR

### training
### prediction
### evaluation

## Contributing



Feel free to help out! [Open an issue](https://github.com/AmsterdamInternships/document-classification-using-large-language-models/issues), submit a [PR](https://github.com/AmsterdamInternships/document-classification-using-large-language-models/pulls)  or [contact us](https://amsterdamintelligence.com/contact/).




## Acknowledgements


This repository was created in collaboration with [Amsterdam Intelligence](https://amsterdamintelligence.com/) for the City of Amsterdam.



Optional: add citation or references here.


## License 

This project is licensed under the terms of the European Union Public License 1.2 (EUPL-1.2).
