import pandas as pd
import gzip
import numpy as np
from multiprocessing import  Pool
import json
import re
import glob
import os
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--MAX_LEN', type=int, default=1000000,help="number of samples stored in a csv file")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

# get all raw files from input folder
raw_file_list = glob.glob("./amazon_raw/*json.gz")
# raw_file_list = ['./amazon_raw/All_Beauty.json.gz','./amazon_raw/Software.json.gz']

# create output folders
out_file_dir = "./amazon_fsp_output/"
if not os.path.exists(out_file_dir):
    os.mkdir(out_file_dir)

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def check_num_sentences(df):
    df['reviewText'] = df['reviewText'].astype(str)
    df['sentences'] = df['reviewText'].apply(lambda x: sent_tokenize(x))
    df['num_sentences'] = df['sentences'].apply(lambda x: len(x))
    return df

def create_sentence_pair(df):
    df['sentence_1'] = df['sentences'].apply(lambda x: (' ').join(x[1:]))
    df['sentence_2'] = df['sentences'].apply(lambda x: x[0])
    # df['sentence_2'] = df['sentence_2'].apply(lambda x: tokenizer.decode(tokenizer.encode(x,max_length=max_length,truncation=True),skip_special_tokens=True))
    return df

def check_sentence_len(df):
    df['len_sent1'] = df['sentence_1'].apply(lambda x: len(x))
    df['is_alphabetic1']= df['sentence_1'].apply(lambda x: bool(re.search('[a-zA-Z]', x)))
    df['len_sent2'] = df['sentence_2'].apply(lambda x: len(x))
    df['is_alphabetic2']= df['sentence_2'].apply(lambda x: bool(re.search('[a-zA-Z]', x)))
    return df

def parallelize_dataframe(df, func, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def process(df_sub, outdir, i):
    df_sub = parallelize_dataframe(df_sub, check_num_sentences)
    df_sub = df_sub[df_sub['num_sentences'] > 2]
    df_sub = parallelize_dataframe(df_sub, create_sentence_pair)
    df_sub = parallelize_dataframe(df_sub, check_sentence_len)
    df_sub = df_sub[(df_sub['len_sent1']>3)&(df_sub['is_alphabetic1']==True)&(df_sub['len_sent2']>3)&(df_sub['is_alphabetic2']==True)] 
    df_sub = df_sub[['sentence_1', 'sentence_2', 'summary', 'overall', 'asin', 'vote']]
    df_sub = df_sub.fillna({'summary':'', 'vote':0})
    df_sub['vote'] = df_sub['vote'].apply(lambda x: int(str(x).replace(',', '')))
    df_sub = df_sub.drop_duplicates()
    print(f'writing into {outdir}/part_{i}.csv')
    df_sub.to_csv(f'{outdir}/part_{i}.csv', index=False)
    
for raw_file_path in raw_file_list:
    print(f'getting data from {raw_file_path}')
    directory = raw_file_path.split('/')[-1].split('.')[0]
    outdir = os.path.join(out_file_dir, directory)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    count = 0
    i = 0
    df = {}
    for d in parse(raw_file_path):
        df[count] = d
        count += 1
        if count==args.MAX_LEN:
            df = pd.DataFrame.from_dict(df, orient='index')
            process(df, outdir, i)
            count=0
            i+=1
            df = {}
    df = pd.DataFrame.from_dict(df, orient='index')
    process(df, outdir, i)

