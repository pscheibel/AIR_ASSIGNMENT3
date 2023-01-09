# AIR_ASSIGNMENT3
Click [here](https://pscheibel.github.io/AIR_ASSIGNMENT3/) to go to our website. 

Install packages for project with "pip install -r requirements.txt"

To add new pdfs for classification, add them to the input folder

Specific queries can be added in the queries folder, an example is given in the queries.txt
A query consists out of a scientific category and a noun, that will be searched for.
Documents that belong to this scientific category (according to our classification model) and contain the noun will be listed in the "queriesResult.txt" file.

There are 4 flags (CACHING_FILES_ENABLED, DATASET_CREATION_ENABLED, TRAINING_ENABLED, INPUT_PREPARATION_ENABLED) in the main.py file:

CACHING_FILES_ENABLED: will cache pdf files to reduce the time needed to execute the script after the first invocation.
DATASET_CREATION_ENABLED: will create the database of pdfs.  
TRAINING_ENABLED: if training was already done, you can disable this to reuse learned model.  
INPUT_PREPARATION_ENABLED: if enabled, this will look for a noun database and use them as cache.
