patience=100
repetition=1
add_hard=0
add_neutral=1
scheduler=linear
max_length=512
seed=1
optimizer=AdamW 
max_label_num=15
option_type=str
save_model=1
notrain=0
gradient_accumulation_steps=1
gradient_checkpointing=0
train_loader_regen=1
out=results.csv
pad_labels=1
lr=1e-5
sent2_length=512
list_for_test=all

echo starts training........................................................................................
# model=roberta-large
model=roberta-base
script=SSTuning.py

input_file=./data_training/amazon_fsp_output_sample500000,./data_training/wiki_fsp_output_neg10_5perArticle
log_folder=./LOGS

n_samples=100000
n_samples_eval=3200
n_samples_test=320
num_steps_for_evaluation=10
epochs=1
BS=32
lr=0.5e-5

list_for_test=yahoo_topics,agnews,dbpedia,20newsgroup_fillna,sst2,imdb,yelp_polarity,MR_sep,amazon_polarity,sst5
list_label_mode_for_test=0
max_label_num=20
max_label_num_nonPAD=10
option_type=str
add_hard=10
save_model=1

for seed in 42; do
accelerate launch --multi_gpu --mixed_precision=fp16 $script \
          --seed $seed --model $model --BS $BS --lr $lr --epochs $epochs\
          --num_steps_for_evaluation $num_steps_for_evaluation --patience $patience \
          --log_folder $log_folder --out $out --save_model $save_model --n_samples $n_samples \
          --add_hard $add_hard --add_neutral $add_neutral --repetition $repetition --max_label_num $max_label_num \
          --scheduler $scheduler --notrain $notrain --pad_labels $pad_labels --n_samples_eval $n_samples_eval\
          --option_type $option_type --input_file $input_file --n_samples_test $n_samples_test --max_length $max_length \
          --optimizer $optimizer --gradient_accumulation_steps $gradient_accumulation_steps \
          --gradient_checkpointing $gradient_checkpointing --train_loader_regen $train_loader_regen \
          --sent2_length $sent2_length --list_for_test $list_for_test --list_label_mode_for_test $list_label_mode_for_test \
          --max_label_num_nonPAD $max_label_num_nonPAD --option_type_test $option_type
done






