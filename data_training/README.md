# Training data generation

* Download datasets
Amazon product review: download from https://nijianmo.github.io/amazon/
Wikipedia: can be downloaded directly from HuggingFace (please check data_prepare_wiki.py)

* Prepare datasets (basic processing)
    * data_prepare_amazon.py
    * data_prepare_wiki.py

* Process datasets (generate final training and validation datasets)
    * data_processing_amazon.py
    * data_processing_wiki.py

* Combined scripts
    * 01_gen_data_wiki.sh
    * 02_gen_data_amazon.sh