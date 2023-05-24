echo step 1, prepare dataset
python data_prepare_wiki.py --num_articles 0 --MAX_LEN 1000000

echo step 2, process dataset
python data_processing_wiki.py --num_neg 10 --num_perArticle 5 --sent2_length 512