import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import datasets
from datasets import load_dataset
import torch, glob
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW 
from multiprocessing import  Pool
from functions import *
import os, time, string, random, logging
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,help="random seed for initialization")
parser.add_argument('--num_neg', type=int, default=0,help="number of maximum hard negatives sampled for each sample")
parser.add_argument('--n_sample', type=int, default=10000,help="number of samples selected from each product category")
parser.add_argument('--sent2_length', type=int, default=512,help="The maximum tokens for each options")
args = parser.parse_args()

logger = setup_logger('general_logger', f'./log_amazon_sampling.log')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class dataset_wrapper_FSP: 
    def __init__(self, data_file, tokenizer,sent2_length=512, num_sample=0,num_neg = 0, num_workers=4):
        dataset = load_dataset("csv", data_files=data_file,num_proc=num_workers)
        print('dataset successfully loaded')
        dataset = dataset['train']
        dataset = dataset.rename_column("sentence_1", "sentence1")
        dataset = dataset.rename_column("sentence_2", "sentence2")
        self.num_sample = num_sample
        self.tokenizer = tokenizer
        self.sent2_length = sent2_length
        self.num_neg = num_neg
        self.total_num = dataset.num_rows
        self.num_workers = num_workers
        
        self.dataset = self.gen_dataset(dataset)   
    
    def gen_dataset(self, dataset):
        if self.num_sample and self.num_sample < self.total_num:
            dataset_gen = dataset.select(np.random.choice(self.total_num, size=self.num_sample, replace=False))
        else: 
            dataset_gen = dataset
        dataset_gen = dataset_gen.filter(lambda ex: ex['sentence2'] != None)
        list_negatives = list(set(dataset_gen['sentence2']))
        if self.num_neg > 0:
            dataset_gen = dataset_gen.map(lambda ex: {'hard_neg': self.get_random_neg_types(ex['sentence2'], self.num_neg, list_negatives)},num_proc=8)
        if self.sent2_length < 512:
            dataset_gen = dataset_gen.map(self.truncat_sent2,num_proc=self.num_workers)
        return dataset_gen
    
    # The following are help functions
    def truncat_sent2(self,example): 
        example['sentence2'] = self.tokenizer.decode(self.tokenizer.encode(example['sentence2'],max_length=self.sent2_length,truncation=True),skip_special_tokens=True)
        return example

    def get_random_neg_types(self, sentence2, num_neg, list_negatives):
        while True:           
            ind_list = np.random.choice(len(list_negatives),size=num_neg,replace=True).tolist() # It is quite slow if replace=Flase
            neg_types = [list_negatives[i] for i in ind_list]
            neg_types_lower = [x.lower for x in neg_types]
            if sentence2.lower() not in neg_types_lower:
                # print(ind_list,sentence2,neg_types)
                return neg_types            
            
MODEL = '/mnt/workspace/models/bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL,use_fast=True)

logger.info('.........Start loading the data........................................................')
n_sample=args.n_sample
for version in ['amazon_fsp_output']:
    logger.info(f'processing: {version}')
    data_folders = glob.glob(f'./{version}/*')
    logger.info(data_folders)

    logger.info(f'processing n_sample = {n_sample}')
    list_dataset = []
    for data_folder in data_folders:
        logger.info(f'Processing: {data_folder}')
        start1 = time.time()
        data_file = glob.glob(data_folder+'/*')
        data_generator = dataset_wrapper_FSP(data_file, tokenizer, sent2_length=args.sent2_length,num_sample=n_sample,num_neg=args.num_neg, num_workers = 8)
        logger.info(f'{data_file[-1].split("/")[-2]}')
        logger.info(f'loading time :{round(time.time()-start1,1)}s, total num: {data_generator.total_num},total num sampled: {data_generator.dataset.num_rows}')

        dataset = data_generator.dataset
        list_dataset.append(dataset)

    dataset_whole = datasets.concatenate_datasets(list_dataset, axis=0)
    logger.info(dataset_whole)
    dataset_whole = dataset_whole.train_test_split(test_size=min(100000,int(0.1*dataset_whole.num_rows)), seed=args.seed)
    dataset_whole.save_to_disk(f'./{version}_sample{n_sample}')
    logger.info(dataset_whole)
    logger.info(f'Finished sampling {version} n_sample = {n_sample}\n')
        
logger.info('Finished sampling\n\n')