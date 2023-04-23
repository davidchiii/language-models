# |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
# |          fine tune          |
# |_____________________________|


import streamlit as st
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import emoji
import time
from PIL import Image
import pandas as pd
import numpy as np
import re


from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

model_name = 'distilbert-base-uncased'

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    learning_rate=0.00005,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)


# prep dataset (load csv)
#   1. load dataset
#   2. clean dataset
#   3. separate into
# load pretrained token izer, cal with dataset, encoding
# build pytorch dataset
# load pretrained model
# load huggingface trainer


class tweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# credits to @yashj302 on medium
# https://medium.com/@yashj302/text-cleaning-using-regex-python-f1dded1ac5bd
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'scuse', 'excuse', text)  # lol
    text = re.sub(r'http[s]?\://\S+', "", text)  # remove http://medium.com
    text = re.sub(r'\s+', ' ', text)  # remove 'VERY   EXTRA   SPACE        '
    text = re.sub(r'[0-9]', "", text)  # remove numbers
    text = re.sub(r'[^\w]', ' ', text)  # remove characters
    text = re.sub(r' +', ' ', text)
    text = text.strip(' ')
    return text


# test
# print(clean_text('oasdfn12312351ahttps://omg.comsfds\n\n\n\n'))


# labels
label_cols = ['toxic', 'severe_toxic', 'obscene',
              'threat', 'insult', 'identity_hate']
# read csv
df_train = pd.read_csv('data/train.csv')
df_test_labels = pd.read_csv('data/test_labels.csv')
df_test = pd.read_csv('data/test.csv')


# drop ids
df_train.drop('id', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)
df_test_labels.drop('id', axis=1, inplace=True)
# clean text
df_train['comment_text'] = df_train['comment_text'].apply(clean_text)
df_test['comment_text'] = df_test['comment_text'].apply(clean_text)


print(df_train.head())
print(df_test_labels.head())
print(df_test.head())


tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )
