# Download datasets for evaluation
To download the datasets shown in the paper, just run: 

```
wget https://huggingface.co/datasets/DAMO-NLP-SG/SSTuning-datasets/resolve/main/data_testing.zip
unzip data_testing.zip
mv data_testing/* .
```

# To test on your own datasets
1. convert the dataset to csv file following the format of the example.
2. modify the the json file to include the converted labels. 