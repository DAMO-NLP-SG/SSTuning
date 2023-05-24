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
parser.add_argument('--seed', type=int, default=42,help="random seed for initialization")
parser.add_argument('--model', type=str, default='roberta-base', help='model for training and evaluation')
parser.add_argument('--epochs', type=int, default=1,help="num of training epochs")
parser.add_argument('--num_steps_for_evaluation', type=int, default=500,help="num of steps for evalustion")
parser.add_argument('--lr', type=float, default=1e-5,help="learning rate")
parser.add_argument('--BS', type=int, default=32,help="batch size")
parser.add_argument('--scheduler', type=str, default='linear',help="lr scheduler")
parser.add_argument('--optimizer', type=str, default='AdamW',choices=['Adam','AdamW'],help="optimizer")
parser.add_argument('--patience', type=int, default=10,help="patience for early stopping")
parser.add_argument('--max_length', type=int, default=512,help="tokenization max_length")
parser.add_argument('--n_samples', type=int, default=10000,help="num of samples per dataset")
parser.add_argument('--n_samples_eval', type=int, default=16000,help="num of samples per dataset")
parser.add_argument('--log_folder', type=str, default='LOGS/log_folder',  help='log file name')    
parser.add_argument('--log', type=str, default='None',  help='log file name')                  
parser.add_argument('--out', type=str, default='results.csv',  help='log file name')
parser.add_argument('--notrain', type=int, default=0, help='train and save the model')
parser.add_argument('--notest', action = 'store_true', help='load and use the model for testing')                  
parser.add_argument('--parallel', action = 'store_true', help='enable multi-GPU training')
parser.add_argument('--use_amp', type = bool, default = True, help='enable Automatic Mixed Precision')
parser.add_argument('--save_model', type=int, default=0, help='save the trained model')
parser.add_argument('--instruction', type=str, default='None', help='the instruction to add, if "None", no instruction.')
parser.add_argument('--pad_labels', type=int, default=1, help='whether pad labels, 0 means no padding, 1 means padding')
parser.add_argument('--repetition', type=int, default=1, help='number of repetitions for each training sample')
parser.add_argument('--max_label_num', type=int, default=20, help='max number of labels')
parser.add_argument('--add_hard', type=float, default=0, help='add how many hard negative labels, if 0, no hard, if (0,1), potion of labels to be hard, if >=1, num of hard')
parser.add_argument('--add_neutral', type=float, default=1, help='add hard negative labels or not')
parser.add_argument('--option_type', type=str, default='str', help='the type of options, can be str or int or As or hash')
parser.add_argument('--n_samples_test', type=int, default=5000,help="num of samples per dataset for testing. if 0, test all the samples")
parser.add_argument('--input_file', type=str, default='none',help="The input file/folder for training")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,help="num of steps for gradient_accumulation")
parser.add_argument('--gradient_checkpointing', type=int, default=0,help="gradient_checkpointing if set to 1, which save memory but slow down training")
parser.add_argument('--train_loader_regen', type=int, default=1,help="regerate training dataloader for each epoch")
parser.add_argument('--sent2_length', type=int, default=100,help="truncate sentence2 to this length")
parser.add_argument('--list_for_test', type=str, default='all',help="The dataset for testing, seperated by ','")
parser.add_argument('--list_label_mode_for_test', type=str, default='0,1',help="The label_mode for testing, seperated by ','")
parser.add_argument('--max_label_num_nonPAD', type=int, default=15, help='max number of labels')
parser.add_argument('--option_type_test', type=str, default='str', help='the type of options for inference, can be str or int or As or 0s or hash')
parser.add_argument('--json_file_test', type=str, default='./data_testing/label_dict_classification.json', help='the json file which contains the dataset label information')

args = parser.parse_args()

# Creating an instance of the `Accelerator` class, with `gradient_accumulation_steps` argument
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

# Checking if the log folder exists. If not, create the folder with `os.makedirs` method
if not os.path.exists(args.log_folder):
      os.makedirs(args.log_folder, exist_ok = True)

# Determine the `model_name` based on the value of `args.model`. If none of the conditions are met, `model_name` will be set to `"other-model"`
if 'bert-base' in args.model: 
    model_name = 'bert-base-uncased'
elif 'roberta-base' in args.model: 
    model_name = 'roberta-base'
elif 'roberta-large' in args.model: 
    model_name = 'roberta-large'
else: 
    model_name = 'other-model'

# Splitting the file names by comma, and storing it in a list `files`
files = args.input_file.split(',')
files = [f.split('/')[-1] for f in files]
file_name = '_'.join(files)    

# Calculating the effective batch size (`BS_effective`)
BS_effective = args.BS * torch.cuda.device_count() * args.gradient_accumulation_steps
# Multiplying the learning rate (`args.lr`) by the ratio of `BS_effective` over 32
args.lr = args.lr * (BS_effective/32)

# Checking if the flag `notrain` is set and set the log file name accordingly.
if args.notrain: 
    logger = setup_logger('general_logger', f'{args.log_folder}/test_model.log')
else: 
    logger = setup_logger('general_logger', f'{args.log_folder}/{model_name}_label-num{args.max_label_num}_maxNonPAD{args.max_label_num_nonPAD}_sent2-len{args.sent2_length}_{args.lr}_epoch{args.epochs}_pad-{args.pad_labels}_option-{args.option_type}_{file_name}_hard-{args.add_hard}_{args.n_samples}_seed{args.seed}.log')

# Setting the seed value for random number generation in Python's random and numpy libraries, as well as in PyTorch
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTHONHASHSEED'] = str(args.seed)

def model_train(model, list_data_generator, train_loader, val_loader_dic, optim, lr_scheduler,model_folder='check_point'):
    num_steps_for_evaluation = args.num_steps_for_evaluation if (args.num_steps_for_evaluation != 0 and args.num_steps_for_evaluation < len(train_loader)) else len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
  
    best_f1_val, best_loss_val = 0, 1e9
    best_model_wts = copy.deepcopy(model.state_dict())
    
    itr = iter(train_loader)
    best_step = 0
    num_training_steps = len(train_loader) * args.epochs
    progress_bar = tqdm(range(num_training_steps))
    for step in range(num_training_steps):
        model.train()
        with accelerator.accumulate(model):
            batch = next(itr)
            labels = batch['labels']
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optim.step()
            lr_scheduler.step()
            optim.zero_grad()        
        progress_bar.update(1)
        
        # Evaluate the validation set
        if (step+1)%num_steps_for_evaluation == 0: 
            if val_loader_dic != None: 
                loss_dic, f1_dic, df_dic = model_eval(model, val_loader_dic)
                f1_val, loss_val = round(sum(f1_dic.values())/len(f1_dic),4), round(sum(loss_dic.values())/len(loss_dic),4)
                if accelerator.is_main_process:
                    logger.info(f'{step+1}th/{num_training_steps},LR is {lr_scheduler.get_last_lr()}, Eval, f1: {f1_dic},loss: {loss_dic}, f1_avg: {f1_val}, loss_avg: {loss_val}')
                if f1_val > best_f1_val:
                    best_f1_val = f1_val
                    if accelerator.is_main_process:
                        logger.info(f'best results so far at step {step+1}')
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_step = step + 1
                    # Save the checkpoint if the criterias are met.
                    if args.save_model and (step > 10000): 
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        if accelerator.is_main_process:
                            unwrapped_model.save_pretrained(f'{model_folder}')
                            tokenizer.save_pretrained(model_folder)
                early_stopping(f1_val, model)
                
                # If the training does not converge, break the loop.
                if (f1_val < 0.5) and (step > 3000): 
                    break
            
            # Check if the early stopping criteria is met
            if early_stopping.early_stop:
                accelerator.print(f"Early stopping at step {step+1}")
                break
        
        # Check if current step is a multiple of the number of steps in the training set
        if (step+1)%(len(train_loader)) == 0:
            # Check if train_loader_regen flag is set to True and if we haven't reached the maximum number of training steps. 
            # If yes, generate a new training set, which means sampling new negative options for each text.
            if args.train_loader_regen and ((step+1) < num_training_steps):
                tmp_since = time.time()
                list_datasets = [data_generator.gen_dataset(split='train') for data_generator in list_data_generator]
                train_loader = DataLoader(datasets.concatenate_datasets(list_datasets, axis=0), batch_size=args.BS, shuffle=True, num_workers=4,pin_memory=True,drop_last=True)
                train_loader = accelerator.prepare(train_loader)
                tmp_time_elapsed = time.time() - tmp_since
                if accelerator.is_main_process:
                    logger.info(f'Generate a new train dataloader. time_elapsed={tmp_time_elapsed//60:.0f}m{tmp_time_elapsed%60:.0f}s.')
            # Get an iterator for the training data loader
            itr = iter(train_loader)
                                       
    model.load_state_dict(best_model_wts) # Load the best model based on the validation set.
    if accelerator.is_main_process:
        logger.info(f'best step is at {best_step}')
    return model, best_step

def model_eval(model, test_dataloader_dic):
    model.eval()
    loss_dic,f1_dic,df_dic = {},{},{}
    for dataset, test_dataloader in tqdm(test_dataloader_dic.items()): 
        pred_list = []
        label_list = []
        total_loss = 0
        for batch in test_dataloader:
            labels = batch['labels']
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss
            loss = accelerator.gather(loss).mean()
            total_loss = len(labels)*loss.item() + total_loss
            predictions = torch.argmax(logits, dim=-1)
            predictions, labels = accelerator.gather_for_metrics((predictions, batch["labels"]))
            pred_list.append(predictions)
            label_list.append(labels)
            
        pred_list = torch.cat(pred_list)
        label_list = torch.cat(label_list)
        f1, accuracy, precision, recall = score_task(label_list, pred_list)
                        
        loss_dic[dataset] = round(total_loss/len(test_dataloader.dataset),4)
        f1_dic[dataset] = round(f1,4)
        df_dic[dataset] = pd.DataFrame({'label':label_list.tolist(),'pred': pred_list.tolist()})
    return loss_dic, f1_dic, df_dic

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
    logger.info(f'BS: {args.BS}, nGPUs: {torch.cuda.device_count()}, grad_acc: {args.gradient_accumulation_steps}, Effective BS: {BS_effective}')
    logger.info(f'adjusted lr: {args.lr}')

# # generate dataloader
input_files = args.input_file.split(',')
if accelerator.is_main_process:
    logger.info(f'input files: {input_files}')

# Get training and validation dataloaders
if not args.notrain:
    data_file_type = 'csv' if 'csv' in args.input_file else 'arrow'
    list_data_generator = [dataset_wrapper_NSP_bigData_add_hard_v2(inp_file, tokenizer, data_file_type = data_file_type,option_type=args.option_type,\
                       max_label_num=args.max_label_num,max_label_num_nonPAD=args.max_label_num_nonPAD,\
                       pad_labels=args.pad_labels,sent2_length=args.sent2_length,\
                       num_sample=args.n_samples, num_sample_val=args.n_samples_eval,add_hard=args.add_hard) for inp_file in input_files]
    
    if accelerator.is_main_process:
        for data_generator in list_data_generator:
            logger.info(f'dataset info: {data_generator.dataset}')
    
    list_datasets = [data_generator.gen_dataset(split='train') for data_generator in list_data_generator]
    train_loader = DataLoader(datasets.concatenate_datasets(list_datasets, axis=0), batch_size=args.BS, shuffle=True, num_workers=4,pin_memory=True,drop_last=True)
    
    val_loader_dic = {'val'+str(i): DataLoader(data_generator.gen_dataset(split='test'), batch_size=args.BS, shuffle=False, num_workers=4,pin_memory=True) for i, data_generator in enumerate(list_data_generator)}
    val_loader_dic = {k: accelerator.prepare(v) for k, v in val_loader_dic.items()}
else: 
    train_loader = None

def train_test(BS = args.BS, lr = args.lr, optimizer = args.optimizer,train_loader=train_loader):
    since = time.time()
    if accelerator.is_main_process:
        logger.info('*'*50+'start training'+'*'*50)
    
    # Load the model prepare with accelerator
    config = AutoConfig.from_pretrained(args.model,num_labels=args.max_label_num)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config = config)
    if args.gradient_checkpointing: 
        model.gradient_checkpointing_enable()
    model = accelerator.prepare(model)
        
    model_folder = f'{args.log_folder}/checkpoint_{model_name}_label-num{args.max_label_num}_maxNonPAD{args.max_label_num_nonPAD}_sent2_len{args.sent2_length}_{args.lr}_epoch{args.epochs}_pad-{args.pad_labels}_option-{args.option_type}_{file_name}_hard-{args.add_hard}_{args.n_samples}_seed{args.seed}'
    
    # Check the flag "notrain" is set. If not, train the model, otherwise skip training.
    if not args.notrain: 
        if (not os.path.exists(model_folder)) and args.save_model:
            os.makedirs(model_folder, exist_ok = True)
        
        # Set the optimizer and LR scheduler
        optim = Adam(model.parameters(), lr=lr, weight_decay=0.01) if optimizer == 'Adam' else AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        steps = len(train_loader) * args.epochs
        if accelerator.is_main_process:
            logger.info(f'num of steps: {steps}')
        if args.scheduler == 'linear':
            lr_scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps= min(steps//10,1000) , num_training_steps= steps)
        elif args.scheduler == 'cosine':
            lr_scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps= min(steps//10,1000) , num_training_steps= steps)
        optim, train_loader, lr_scheduler = accelerator.prepare(optim, train_loader, lr_scheduler)
        
        if accelerator.is_main_process:
            logger.info(f'Before training, {get_gpu_utilization()}') # Log the GPU memory utilization for reference.
            
        # Train the model and get the best step 
        model, best_step = model_train(model, list_data_generator,train_loader, val_loader_dic = val_loader_dic, optim = optim, \
                                       lr_scheduler = lr_scheduler,model_folder=model_folder)     
        
        time_elapsed=time.time()-since
        if accelerator.is_main_process:
            logger.info(f'time_elapsed={time_elapsed//60:.0f}m{time_elapsed%60:.0f}s.')
            logger.info(f'After training, {get_gpu_utilization()}')

    # Check the flag "notest" is set. If not, test the model on test datasets, otherwise skip testing.
    if not args.notest:
        list_label_mode = [int(x) for x in args.list_label_mode_for_test.split(',')]
        if accelerator.is_main_process:
            logger.info('**************start doing testing*******************************')
        
        # Prepare the data processor for the test datasets
        data_processor_CLS = Data_processor_wrapper_v3(tokenizer,label_num=args.max_label_num,option_type=args.option_type_test,pad_labels=args.pad_labels,
                                                       num_sample=args.n_samples_test,num_workers=4,list_for_test=args.list_for_test,json_file = args.json_file_test)
        # Change the labels if needed
        # for d in ['sst2', 'imdb', 'sent140_2', 'yelp_polarity', 'MR']:
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
                best_step = 0 if args.notrain else best_step
                writer.writerow([args.seed, args.model,'NA0',args.lr,BS_effective,args.epochs, args.optimizer, \
                                 f'time_elapsed={time_elapsed//60:.0f}m{time_elapsed%60:.0f}s.',\
                                 args.input_file,torch.cuda.device_count(),get_gpu_utilization(),args.n_samples,args.BS,args.gradient_accumulation_steps,
                                 args.gradient_checkpointing,args.n_samples_test, best_step,args.pad_labels, args.max_label_num, args.add_hard,\
                                args.sent2_length,args.option_type,args.scheduler,label_mode,args.max_label_num_nonPAD,args.option_type_test,'NA1',\
                                f1_dic['yahoo_topics'],f1_dic['agnews'],f1_dic['dbpedia'],f1_dic['20newsgroup_fillna'],\
                                 f1_dic['sst2'],f1_dic['imdb'],f1_dic['yelp_polarity'],f1_dic['MR_sep'],\
                                 f1_dic['amazon_polarity'],f1_dic['sst5'],sums
                                ]) 
                OUT.close()
    
    # Save the model
    if args.save_model:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(model_folder)
            tokenizer.save_pretrained(model_folder)
            config.save_pretrained(model_folder)

if accelerator.is_main_process:
    print_args(args,logger)

train_test()

if accelerator.is_main_process:
    logger.info('------------------end logging------------------------------------------\n\n')