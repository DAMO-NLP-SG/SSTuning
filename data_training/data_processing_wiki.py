import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from datasets import load_dataset, Dataset, load_from_disk
import torch, glob
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW 
from functions import *
import os, time, string, random, logging, datasets
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,help="random seed for initialization")
parser.add_argument('--num_neg', type=int, default=10,help="number of maximum hard negatives sampled for each sample")
parser.add_argument('--num_perArticle', type=int, default=5,help="number of samples selected from each article")
parser.add_argument('--sent2_length', type=int, default=512,help="The maximum tokens for each options")
args = parser.parse_args()

logger = setup_logger('general_logger', f'./log_wiki_sampling.log')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

logger.info('.........Start loading the data........................................................')
def process_example(ex,num_neg=30):
    # Remove sentence_First if appear in list_First, then sample list_First upto num_neg
    ex['hard_neg'] = list(set(ex['hard_neg']))
    if ex['sentence_2'] in ex['hard_neg']: 
        ex['hard_neg'].remove(ex['sentence_2'])
    if len(ex['hard_neg']) > num_neg: 
        ex['hard_neg'] = random.sample(ex['hard_neg'],num_neg)
    return ex

num_neg = args.num_neg
num_perArticle = args.num_perArticle
sent2_length=args.sent2_length

list_dataset_sample = []
folder = 'wiki_fsp_output'
data_files = glob.glob(f'./{folder}/part_*.csv')
for i in range(len(data_files)):
    start2 = time.time()
    
    num_workers = 1
    data_file = data_files[i]
    logger.info(f'processing {data_file}....')
    df = pd.read_csv(data_file)
    df = df.drop_duplicates(subset=['sentence_1','sentence_2']).reset_index(drop=True)
    if sent2_length < 512:
        df['sentence_2'] = df['sentence_2'].progress_apply(lambda x: tokenizer.decode(tokenizer.encode(x,max_length=sent2_length,truncation=True),skip_special_tokens=True))
    df_hard = df.groupby('article_title')['sentence_2'].apply(list).reset_index(name='hard_neg')
    df = pd.merge(df, df_hard, how='inner')
    
    logger.info(f'generating the dataset for part_{i}')
    ds_FSP = Dataset.from_pandas(df,preserve_index=False)
    ds_FSP = ds_FSP.map(lambda ex: process_example(ex,num_neg=num_neg),num_proc=num_workers)
    ds_FSP = ds_FSP.rename_column('sentence_2','sentence2')
    ds_FSP = ds_FSP.rename_column('sentence_1','sentence1')
    
    logger.info(f'generating sampled dataset for part_{i}')
    df = ds_FSP.to_pandas()
    df = df.groupby('article_title').sample(num_perArticle, replace=True).drop_duplicates(subset=['sentence1','sentence2']).reset_index(drop=True)
    ds_sample = Dataset.from_pandas(df,preserve_index=False)
    list_dataset_sample.append(ds_sample)
    
    end2 = time.time()
    logger.info(f'The data file is {data_file}, num_workers: {num_workers}, data gen time: {end2-start2}s \n, dataset is {ds_FSP}')
    logger.info(f'The sampled dataset is \n {ds_sample}')

dataset_whole = datasets.concatenate_datasets(list_dataset_sample, axis=0)
dataset_whole = dataset_whole.train_test_split(test_size=min(100000,int(0.1*dataset_whole.num_rows)), seed=args.seed)
logger.info(dataset_whole)
dataset_whole.save_to_disk(f'./{folder}_neg{num_neg}_{num_perArticle}perArticle')

logger.info('\n\n')