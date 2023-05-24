import numpy as np
import pandas as pd
import torch, string, random
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import sklearn.metrics as metrics
import logging, sys
from typing import Optional
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import json, time
from collections import defaultdict
from pynvml import *
from datasets import load_dataset, ClassLabel, load_from_disk

def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"GPU memory occupied: {info.used//1024**2} MB."

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def print_args(args,logger, path=None):
	if path:
		output_file = open(path, 'w')
	# logger = logging.getLogger(__name__)
	logger.info("Arguments:")
	args.command = ' '.join(sys.argv)
	items = vars(args)
	# for key in sorted(list(items.keys()), key=lambda s: s.lower()):
	for key in list(items.keys()):
		value = items[key]
		if not value:
			value = "None"
		logger.info("  " + key + ": " + str(items[key]))
		if path is not None:
			output_file.write("  " + key + ": " + str(items[key]) + "\n")
	if path:
		output_file.close()
	del args.command

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        #score = -val_loss
        # set it as + when early stop accoding to accuracy.
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            #self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.trace_func(f'Validation F1 increase ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
class dataset_wrapper_NSP_bigData_add_hard_v2: 
    # this class generate classifition dataset from NLI dataset
    # v2: add the option to control max_label_num_nonPAD and change the range of add_hard (number of hard negatives, or percentage of hard negatives)
    def __init__(self, data_file, tokenizer,data_file_type = 'arrow',option_type = 'str',pad_labels=1,max_label_num=15, max_label_num_nonPAD=15,sent2_length=512, num_sample=0,num_sample_val=10000,add_hard=0,num_workers=4):
        if data_file_type == 'csv':
            dataset = load_dataset("csv", data_files=data_file,num_proc=num_workers)
            dataset = dataset['train']
            dataset = dataset.rename_column("sentence_1", "sentence1")
            dataset = dataset.rename_column("sentence_2", "sentence2")
            dataset = dataset.train_test_split(test_size=min(dataset.num_rows//10,num_sample_val), seed=42)
        else: 
            dataset = load_from_disk(data_file)
        print('dataset successfully loaded')

        self.num_sample = num_sample
        self.tokenizer = tokenizer
        # self.list_ABC = [x for x in string.ascii_uppercase] if option_type == 'str' else [str(x) for x in range(100)]
        if option_type == 'str':
            self.list_ABC = [x for x in string.ascii_uppercase]
        elif option_type == 'int':
            self.list_ABC = [str(x) for x in range(100)]
        elif option_type == 'As':
            self.list_ABC = ['A']*100
        elif option_type == '0s':
            self.list_ABC = ['0']*100
        else: 
            self.list_ABC = ['#']*100
        self.pad_token = tokenizer.pad_token
        self.sep_token = tokenizer.sep_token
        self.pad_labels = pad_labels
        self.max_label_num = max_label_num
        self.max_label_num_nonPAD = max_label_num_nonPAD if max_label_num_nonPAD <= max_label_num else max_label_num
        self.max_length = 512
        self.sent2_length = sent2_length
        self.num_sample_val = num_sample_val
        # self.add_hard = add_hard if ('wiki_full' in data_file or 'cc_news' in data_file) else 0
        self.add_hard = add_hard #if ('wiki_full' in data_file or 'cc_news' in data_file) else 0
        
        self.dataset = dataset
        # print(dataset)
        self.total_num_train = dataset['train'].num_rows
        self.total_num_test = dataset['test'].num_rows
        
        # self.num_workers = len(os.sched_getaffinity(0))
        self.num_workers = min(num_workers * torch.cuda.device_count(),16) if self.total_num_train > 100000 else num_workers        
    
    def gen_dataset(self,split='train'):
        dataset = self.dataset[split]
        if self.num_sample and self.num_sample < self.total_num_train and split=='train':
            dataset = dataset.select(np.random.choice(self.total_num_train, size=self.num_sample, replace=False))
        elif self.num_sample_val and self.num_sample_val < self.total_num_test and split=='test':
            dataset = dataset.select(np.random.choice(self.total_num_test, size=self.num_sample_val, replace=False))
        dataset = dataset.shuffle()
        dataset_gen = self.sample_n_tokenize(dataset)
        return dataset_gen
    
    # The following are help functions
    def sample_n_tokenize(self, dataset):
        if self.sent2_length < 512:
            print('starting truncate sent2')
            dataset = dataset.map(self.truncat_sent2,num_proc=self.num_workers)
        list_negatives = dataset['sentence2']
        dataset_gen = dataset.map(lambda example: self.gen_sample_fast(example, list_negatives),num_proc=4) # Experiment show when set to 4, it is fastest.
        dataset_gen = dataset_gen.map(self.add_embeddings,batched=True,num_proc=self.num_workers)
        dataset_gen = dataset_gen.remove_columns([x for x in dataset_gen.column_names if x not in ['labels', 'input_ids', 'token_type_ids', 'attention_mask']]).with_format("torch")
        return dataset_gen
    
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

    def gen_sample_fast(self, example, list_negatives):
        sent1, sent2 = example['sentence1'], example['sentence2']
        n = random.randint(1,self.max_label_num_nonPAD-1)
        example['n_labels'] = n+1
        
        if self.add_hard > 0 and 'hard_neg' in list(example.keys()):
            example['hard_neg_selected'] = []
            if len(example['hard_neg']) > 0: 
                n_hard = min(n,int(self.add_hard),len(example['hard_neg'])) if self.add_hard >= 1 else min(int(n*self.add_hard),len(example['hard_neg']))
                label_hard = random.sample(example['hard_neg'],n_hard)
                example['hard_neg_selected'] = label_hard
                n = n-n_hard
                list_label = self.get_random_neg_types(sent2, n,list_negatives) if n > 0 else []
                # list_label.append(label_hard)
                list_label.extend(label_hard)
            else: 
                list_label = self.get_random_neg_types(sent2, n,list_negatives)
        else: 
            list_label = self.get_random_neg_types(sent2, n,list_negatives)
        
        list_label.append(sent2)
            
        if self.pad_labels > 0: 
            list_label = list_label + [self.pad_token]* (self.max_label_num - len(list_label))
        random.shuffle(list_label)
          
        s_option = ' '.join(['('+self.list_ABC[i]+') '+str(list_label[i]) for i in range(len(list_label))]) 
        example['labels'] = list_label.index(sent2)
        example['text'] = f'{s_option} {self.sep_token} {sent1}'
        return example           
    
    def add_embeddings(self,examples):       
        return self.tokenizer(examples["text"],truncation=True, padding='max_length', max_length=self.max_length)             
        
class Data_processor_wrapper_v3:
    def __init__(self,tokenizer,label_num=15,option_type='str',pad_labels=1,max_length=512,num_sample=0,num_workers=4,list_for_test='all',json_file = '../zeroShotDatasets/label_dict_classification.json'):
        self.tokenizer, self.pad_labels, self.label_num  = tokenizer, pad_labels, label_num
        self.pad_token,self.sep_token = tokenizer.pad_token,tokenizer.sep_token
        self.max_length, self.num_sample = max_length, num_sample
        self.num_workers = num_workers
        # self.list_ABC = [x for x in string.ascii_uppercase] if option_type == 'str' else [str(x) for x in range(100)]   
        if option_type == 'str':
            self.list_ABC = [x for x in string.ascii_uppercase]
        elif option_type == 'int':
            self.list_ABC = [str(x) for x in range(100)]
        elif option_type == 'As':
            self.list_ABC = ['A']*100
        elif option_type == '0s':
            self.list_ABC = ['0']*100
        elif option_type == 'BA':
            self.list_ABC = ['B', 'A', 'D', 'C', 'F', 'E', 'H', 'G', 'J', 'I', 'L', 'K', 'N', 'M', 'P', 'O', 'R', 'Q', 'T', 'S', 'V', 'U', 'X', 'W', 'Z', 'Y']
        else: 
            self.list_ABC = ['#']*100
        
        with open(json_file, 'r') as f:
            self.dic_list_label = json.load(f)
        
        # self.list_dataset = ['sst2','imdb','bench_emotion','yahoo','agnews','dbpedia']
        self.list_dataset = list(self.dic_list_label.keys())
        self.list_for_test = self.datasets.keys() if list_for_test=='all' else list_for_test.split(',')

        self.datasets = self.load_datasets()
        
    def load_datasets(self):
        datasets = {}
        for d in self.list_for_test:
            if d in self.list_dataset:
                dataset = load_dataset("csv", data_files=f'./zeroShotDatasets/{d}/test.csv')['train']
                dataset = dataset.rename_column("label","labels")
                # if d == 'bench_emotion':
                #     dataset = dataset.remove_columns(['label_text', 'category'])
                # elif d == 'yahoo':
                #      dataset = dataset.remove_columns("label_text")
                if self.num_sample and self.num_sample < dataset.num_rows: 
                    # dataset = dataset.select(np.random.choice(dataset.num_rows, size=self.num_sample, replace=False))
                    dataset = dataset.shuffle(seed=42).select(range(self.num_sample))
                    # if accelerator.is_main_process:
                    #     logger.info(f'The dataset is:\n {dataset}')
                datasets[d] = dataset
        return datasets
    
    def gen_dataset_labeled(self,label_mode=0,remove_text=True): 
        list_for_test = self.list_for_test
        updated_datasets = {}
        for d in list_for_test:
            if d not in self.datasets.keys():
                continue
            dataset = self.datasets[d]
            # if accelerator.is_main_process:
            #     logger.info(f'The dataset is: {dataset.num_rows}')
            print(f'Generating data for {d}')
            if label_mode < len(self.dic_list_label[d]):
                list_label = self.dic_list_label[d][label_mode] 
            else: 
                list_label = self.dic_list_label[d][0]
                
            # if accelerator.is_main_process:
            #     logger.info(f'The dataset Mark1 is: {dataset}')
                
            dataset = dataset.map(lambda example: {'joined_text': self.add_prefix(example['text'],list_label)})
            
            # if accelerator.is_main_process:
            #     logger.info(f'The dataset Mark2 is: {dataset}')
                
            dataset = dataset.map(self.add_embeddings,batched=True,num_proc=self.num_workers)
            if remove_text:
                dataset = dataset.remove_columns([x for x in dataset.column_names if x not in ['labels', 'input_ids', 'token_type_ids', 'attention_mask']]).with_format("torch")
            new_features = dataset.features.copy()
            new_features["labels"] = ClassLabel(names=list_label)
            dataset = dataset.cast(new_features)
            
            updated_datasets[d] = dataset
        return updated_datasets
                              
    def add_prefix(self,text,list_label):
        if self.pad_labels:
            list_label = list_label + [self.pad_token]* (self.label_num - len(list_label))
        s_option = ' '.join(['('+self.list_ABC[i]+') '+list_label[i] for i in range(len(list_label))])
        return f'{s_option} {self.sep_token} {text}'
    
    def add_embeddings(self,examples):       
        return self.tokenizer(examples["joined_text"],truncation=True, padding='max_length', max_length=self.max_length)   
    
def score_task(labels, preds, task = 'accuracy'):
    # average can be {‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’
    labels = labels.cpu().long()
    preds = preds.cpu().long()
    
    tweeteval_result = -1
    results = metrics.classification_report(labels, preds, output_dict=True,zero_division=0)
    # print(results)
    
    if 'emoji' in task:
        tweeteval_result = results['macro avg']['f1-score'] 
          
    elif 'sst5' in task: 
        recalls = [results[str(x)]['recall'] for x in range(5)]
        tweeteval_result = sum(recalls)/len(recalls)
       
    else: 
        tweeteval_result = results['accuracy']
    
    accuracy = results['accuracy']
    precision = results['macro avg']['precision']
    recall = results['macro avg']['recall']
                                    
    return tweeteval_result, accuracy, precision, recall