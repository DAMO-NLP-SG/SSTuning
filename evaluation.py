import pandas as pd
import numpy as np
import torch,random, logging,string,re,argparse
import os, copy, csv, sys, time, math, glob
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tqdm import tqdm
tqdm.pandas()  
from functions import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModel, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, Adafactor
from torch.optim import Adam, AdamW 
import json, datasets
from collections import defaultdict
from sklearn.model_selection import train_test_split
from accelerate import Accelerator

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='roberta-base', help='model for training and evaluation')
parser.add_argument('--num_steps_for_evaluation', type=int, default=500,help="num of steps for evalustion")
parser.add_argument('--BS', type=int, default=32,help="batch size")
parser.add_argument('--log_folder', type=str, default='LOGS/log_folder',  help='log file name')    
parser.add_argument('--log', type=str, default='test_model.log',  help='log file name')                  
parser.add_argument('--out', type=str, default='results.csv',  help='log file name')
parser.add_argument('--max_label_num', type=int, default=20, help='max number of labels')
parser.add_argument('--n_samples_test', type=int, default=5000,help="num of samples per dataset for testing. if 0, test all the samples")
parser.add_argument('--list_for_test', type=str, default='all',help="The dataset for testing, seperated by ','")
parser.add_argument('--list_label_mode_for_test', type=str, default='0,1',help="The label_mode for testing, seperated by ','")
parser.add_argument('--option_type_test', type=str, default='str', help='the type of options for inference, can be str or int or As or 0s or hash')
parser.add_argument('--json_file_test', type=str, default='./data_testing/label_dict_classification.json', help='the json file which contains the dataset label information')

args = parser.parse_args()

# Creating an instance of the `Accelerator` class
accelerator = Accelerator()

# Checking if the log folder exists. If not, create the folder with `os.makedirs` method
if not os.path.exists(args.log_folder):
      os.makedirs(args.log_folder, exist_ok = True)

# Calculating the effective batch size (`BS_effective`)
BS_effective = args.BS * torch.cuda.device_count() 
logger = setup_logger('general_logger', f'{args.log_folder}/{args.log}')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def model_test(model, test_dataloader_dic):
    model.eval()
    f1_dic,df_dic = {},{} # Initialize the two dictionaries to be returned

    for dataset, test_dataloader in test_dataloader_dic.items():
        pred_list = []
        label_list = []
        num_classes = test_dataloader.dataset.features['labels'].num_classes
        if num_classes > args.max_label_num:
            if accelerator.is_main_process:
                logger.info(f'For {dataset}, it has {num_classes} classes, which is larger than max allowed label num {args.max_label_num}, thus skip.')
            continue
        for batch in tqdm(test_dataloader):
            labels = batch['labels']
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits[:,0:num_classes], dim=-1)
            predictions, labels = accelerator.gather_for_metrics((predictions, batch["labels"]))
            pred_list.append(predictions)
            label_list.append(labels)
            
        pred_list = torch.cat(pred_list)
        label_list = torch.cat(label_list)
        f1, accuracy, precision, recall = score_task(label_list, pred_list)
                        
        f1_dic[dataset] = round(f1,4)
        df_dic[dataset] = pd.DataFrame({'label':label_list.tolist(),'pred': pred_list.tolist()})
    return f1_dic, df_dic

if accelerator.is_main_process:
    logger.info('------------------start logging-------------------------------------')
MODEL = args.model
tokenizer = AutoTokenizer.from_pretrained(args.model)
pad_token, sep_token = tokenizer.pad_token, tokenizer.sep_token
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if accelerator.is_main_process:
    logger.info(f'num of GPU available {torch.cuda.device_count()}')
    logger.info(f'num of CPU available {len(os.sched_getaffinity(0))}')
    logger.info(f'BS: {args.BS}, nGPUs: {torch.cuda.device_count()}, grad_acc:  Effective BS: {BS_effective}')


def train_test(BS = args.BS):
    since = time.time()
    if accelerator.is_main_process:
        logger.info('*'*50+'start training'+'*'*50)
    
    # Load the model prepare with accelerator
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model = accelerator.prepare(model)

    list_label_mode = [int(x) for x in args.list_label_mode_for_test.split(',')]
    if accelerator.is_main_process:
        logger.info('**************start doing testing*******************************')

    # Prepare the data processor for the test datasets
    data_processor_CLS = Data_processor_wrapper_v3(tokenizer,label_num=args.max_label_num,option_type=args.option_type_test,pad_labels=1,
                                                   num_sample=args.n_samples_test,num_workers=4,list_for_test=args.list_for_test,json_file = args.json_file_test)
    # Change the labels if needed
    # for d in ['sst2', 'imdb', 'yelp_polarity', 'MR']:
    #     data_processor_CLS.dic_list_label[d] =[["It's terrible.","It's great."],['negative.','positive.'],["terrible.","great."],['bad','good'],\
    #                                            ["awful.","great."],['terrible','awesome'],['horrible','incredible'],\
    #                                             ['terrible,bad,awful,horrible','good,great,awesome,incredible']]

    if accelerator.is_main_process:
        logger.info(f'Available datasets for testing CLS: {data_processor_CLS.list_dataset}')
    list_dataset_whole = data_processor_CLS.list_dataset 
    dic_list_label_whole = data_processor_CLS.dic_list_label

    # For each version of labels, reformulate the test datasets and test the model    
    results = []
    for label_mode in list_label_mode:
        f1_dic = {k: 0 for k in list_dataset_whole}
        test_datasets = data_processor_CLS.gen_dataset_labeled(label_mode=label_mode)
        list_num_rows = {k: v.num_rows for k, v in test_datasets.items()}
        test_loader_dic = {k: DataLoader(v, batch_size=args.BS, shuffle=False, num_workers=4,pin_memory=True) for k, v in test_datasets.items()}
        test_loader_dic = {k: accelerator.prepare(v) for k, v in test_loader_dic.items()}
        f1_dic_tmp, df_dic = model_test(model, test_loader_dic)  
        f1_dic_tmp = {k: round(v*100,2) for k, v in f1_dic_tmp.items()}
        for k, v in f1_dic_tmp.items():
            f1_dic[k] = f1_dic_tmp[k]
        sums = round(sum(f1_dic.values()),2)
        results.append(f1_dic)

        time_elapsed=time.time()-since
        if accelerator.is_main_process:
            for k in f1_dic_tmp.keys(): 
                if k in dic_list_label_whole.keys():
                    if label_mode < len(dic_list_label_whole[k]):
                        logger.info(f'{k},{dic_list_label_whole[k][label_mode]}')
                    else: 
                        logger.info(f'{k},{dic_list_label_whole[k][0]}')
            logger.info(f'Num of samples for testing: {list_num_rows}')
            logger.info(get_gpu_utilization())
            logger.info(f'time_elapsed={time_elapsed//60:.0f}m{time_elapsed%60:.0f}s.')
            logger.info(f'Testing mode {label_mode}: {f1_dic_tmp}, sum: {sum(f1_dic.values())}')
            logger.info('-'*100)

            # write the results to csv
            OUT = open(args.log_folder+'/'+args.out, 'a', newline='')
            writer = csv.writer(OUT)
            writer.writerow([args.model,args.BS,torch.cuda.device_count(),BS_effective, f'time_elapsed={time_elapsed//60:.0f}m{time_elapsed%60:.0f}s.',\
                             get_gpu_utilization(),args.n_samples_test,label_mode,' '.join(f1_dic.keys())]+\
                            list(f1_dic.values())+[sums])
            OUT.close()
    

if accelerator.is_main_process:
    print_args(args,logger)

train_test()

if accelerator.is_main_process:
    logger.info('------------------end logging------------------------------------------\n\n')