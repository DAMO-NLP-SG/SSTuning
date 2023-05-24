from datasets import *
import csv
import os
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_articles', type=int, default=0,help="number of articles to process, if 0, process all the articles")
parser.add_argument('--MAX_LEN', type=int, default=1000000,help="number of samples stored in a csv file")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

# # load dataset from hugging face
# dataset = load_dataset("wikipedia", "20220301.en")
# dataset.save_to_disk('./wiki_raw')

# load dataset from disk
dataset = load_from_disk('./wiki_raw')

# set output path
output_folder = 'wiki_fsp_output'
os.mkdir(os.path.join(os.getcwd(),output_folder))

# process data based on FSP
k = 0
count = 0
rows = []
num_articles = len(dataset['train']) if args.num_articles ==0 else args.num_articles
print(f'Total number of articles: {num_articles}.')
fields = ['sentence_1', 'sentence_2', 'article_title']
for j in tqdm(range(num_articles)):
# for j in tqdm(range(10000)):
    raw_text = dataset['train'][j]['text']
    title = dataset['train'][j]['title']
    paragraphs = raw_text.replace('\n\n', '\n').split('\n')
    for paragraph in paragraphs:
        if '.' not in paragraph:
            continue
        sentences = sent_tokenize(paragraph)
        num_sent = len(sentences)
        if num_sent==1:
            continue
        sentence_1 = (' ').join(sentences[1:])
        sentence_2 = sentences[0]
        if ('.' not in sentence_1) \
        or ('.' not in sentence_2) \
        or (len(sentence_1)<10) \
        or (len(sentence_2)<10) \
        or (len(sentence_1.split())<3) \
        or (len(sentence_2.split())<3):
            continue
        row = [sentence_1, sentence_2, title]
        rows.append(row)
        count += 1
        if count==args.MAX_LEN:
            print(f'writing part {k}')
            with open(f'{output_folder}/part_{k}.csv', 'w') as f:
                write = csv.writer(f)
                write.writerow(fields)
                write.writerows(rows)
                k+=1
                count=0
                rows=[]
             
            
# write the remaining part
with open(f'{output_folder}/part_{k}.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows)

