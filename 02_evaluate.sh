echo starts training........................................................................................
model=DAMO-NLP-SG/zero-shot-classify-SSTuning-base
# model=DAMO-NLP-SG/zero-shot-classify-SSTuning-large
# model=DAMO-NLP-SG/zero-shot-classify-SSTuning-ALBERT

script=evaluation.py

log_folder=./LOGS
out=results.csv

json_file_test=./data_testing/label_dict_classification.json
list_for_test=yahoo_topics,agnews,dbpedia,20newsgroup_fillna,sst2,imdb,yelp_polarity,MR_sep,amazon_polarity,sst5

# json_file_test=./data_testing/label_dict_classification_1k.json
# list_for_test=yahoo_topics_1k,agnews_1k,dbpedia_1k,20newsgroup_fillna_1k,sst2_1k,imdb_1k,yelp_polarity_1k,MR_sep_1k,amazon_polarity_1k,sst5_1k

list_label_mode_for_test=0
n_samples_test=1000
BS=32

accelerate launch --multi_gpu --mixed_precision=fp16 $script \
          --model $model --BS $BS --n_samples_test $n_samples_test \
          --log_folder $log_folder --out $out \
          --list_for_test $list_for_test --list_label_mode_for_test $list_label_mode_for_test \
          --json_file_test $json_file_test
