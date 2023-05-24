echo starts training........................................................................................
# model=roberta-large
model=roberta-base
model=DAMO-NLP-SG/zero-shot-classify-SSTuning-base
script=SSTuning.py
notrain=1

input_file=./data_training/amazon_fsp_output_sample500000,./data_training/wiki_fsp_output_neg10_5perArticle
log_folder=./LOGS
out=results.csv

# list_for_test=yahoo_topics,agnews,dbpedia,20newsgroup_fillna,sst2,imdb,yelp_polarity,MR_sep,amazon_polarity,sst5
list_for_test=agnews,sst2
list_label_mode_for_test=0
n_samples_test=320
BS=32

accelerate launch --multi_gpu --mixed_precision=fp16 $script \
          --model $model --BS $BS --n_samples_test $n_samples_test \
          --log_folder $log_folder --out $out --notrain $notrain \
          --list_for_test $list_for_test --list_label_mode_for_test $list_label_mode_for_test \
          





