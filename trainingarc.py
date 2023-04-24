# |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
# |          fine tune          |
# |_____________________________|


from torch import cuda
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
from tqdm import tqdm


from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

import numpy as np
import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from torch import cuda

model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

device = 'cuda' if cuda.is_available() else 'cpu'

EPOCHS = 1
LEARNING_RATE = 1e-05

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
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

class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


model = DistilBertTokenizerFast.from_pretrained(model_name, num_labels=5, problem_type='multi_label_classification')

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

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
df_train = pd.read_csv('data/train.csv')
df_train.drop('id', axis=1, inplace=True)
df_train['comment_text'] = df_train['comment_text'].apply(clean_text)
df_train['labels'] = df_train[label_cols].apply(lambda x: list(x), axis=1)


# df_train['tokens'] = [tokenizer(a, max_length=512, truncation=True)
#                       for a in tqdm(df_train['comment_text'].values)]

df_train.drop(label_cols, axis=1, inplace=True)





# model = DistilBertForSequenceClassification.from_pretrained(model_name)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )
