# Zero-shot text classification trained with self-supervised tuning
This repository contains the code and pre-trained models for for ACL paper "Zero-Shot Text Classification via Self-Supervised Tuning".

## Model description
The model is tuned with unlabeled data using a learning objective called first sentence prediction (FSP). 
The FSP task is designed by considering both the nature of the unlabeled corpus and the input/output format of classification tasks. 
The training and validation sets are constructed from the unlabeled corpus using FSP. 

![](./figures/SSTuning.png)

During tuning, BERT-like pre-trained masked language models such as RoBERTa and ALBERT are employed as the backbone, and an output layer for classification is added. 
The learning objective for FSP is to predict the index of the correct label. 
A cross-entropy loss is used for tuning the model.

## Model variations
There are three versions of models released. The details are: 

| Model | Backbone | #params | accuracy | Speed | #Training data
|------------|-----------|----------|-------|-------|----|
|   [zero-shot-classify-SSTuning-base](https://huggingface.co/DAMO-NLP-SG/zero-shot-classify-SSTuning-base)    |  [roberta-base](https://huggingface.co/roberta-base)      |  125M    |  Low    |  High    | 20.48M |  
|   [zero-shot-classify-SSTuning-large](https://huggingface.co/DAMO-NLP-SG/zero-shot-classify-SSTuning-large)    |    [roberta-large](https://huggingface.co/roberta-large)      | 355M     |   Medium   | Medium | 5.12M |
|   [zero-shot-classify-SSTuning-ALBERT](https://huggingface.co/DAMO-NLP-SG/zero-shot-classify-SSTuning-ALBERT)   |  [albert-xxlarge-v2](https://huggingface.co/albert-xxlarge-v2)      |  235M   |    High  | Low| 5.12M |

Please note that zero-shot-classify-SSTuning-base is trained with more data (20.48M) than the paper, as this will increase the accuracy.


## Intended uses & limitations
The model can be used for zero-shot text classification such as sentiment analysis and topic classification. No further finetuning is needed.

The number of labels should be 2 ~ 20. 

### How to use
You can try the model with the Colab [Notebook](https://colab.research.google.com/drive/17bqc8cXFF-wDmZ0o8j7sbrQB9Cq7Gowr?usp=sharing).

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, string, random

tokenizer = AutoTokenizer.from_pretrained("DAMO-NLP-SG/zero-shot-classify-SSTuning-base")
model = AutoModelForSequenceClassification.from_pretrained("DAMO-NLP-SG/zero-shot-classify-SSTuning-base")

text = "I love this place! The food is always so fresh and delicious."
list_label = ["negative", "positive"]

list_ABC = [x for x in string.ascii_uppercase]
def add_prefix(text, list_label, shuffle = False):
    list_label = [x+'.' if x[-1] != '.' else x for x in list_label]
    list_label_new = list_label + [tokenizer.pad_token]* (20 - len(list_label))
    if shuffle: 
        random.shuffle(list_label_new)
    s_option = ' '.join(['('+list_ABC[i]+') '+list_label_new[i] for i in range(len(list_label_new))])
    return f'{s_option} {tokenizer.sep_token} {text}', list_label_new

text_new, list_label_new = add_prefix(text,list_label,shuffle=False)

encoding = tokenizer([text_new],truncation=True, padding='max_length',max_length=512, return_tensors='pt')
with torch.no_grad():
    logits = model(**encoding).logits
    probs = torch.nn.functional.softmax(logits, dim = -1).tolist()
    predictions = torch.argmax(logits, dim=-1)

print(probs)
print(predictions)
```


### BibTeX entry and citation info
```bibtxt
@inproceedings{acl23/SSTuning,
  author    = {Chaoqun Liu and
               Wenxuan Zhang and
               Guizhen Chen and
               Xiaobao Wu and
               Anh Tuan Luu and
               Chip Hong Chang and 
               Lidong Bing},
  title     = {Zero-Shot Text Classification via Self-Supervised Tuning},
  booktitle = {Findings of the 2023 ACL},
  year      = {2023},
  url       = {},
}
```
