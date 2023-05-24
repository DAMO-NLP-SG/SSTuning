echo step 1, prepare dataset
python data_prepare_amazon.py --MAX_LEN 1000000

echo step 2, process dataset
python data_processing_amazon.py --n_sample 500000 --sent2_length 512