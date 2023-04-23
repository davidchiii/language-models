import streamlit as st
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
import emoji
import time
from PIL import Image


model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name)  # not used
pl = pipeline("sentiment-analysis", model=model,
              tokenizer=tokenizer, framework='pt')
results = pl(
    ["We are very happy to show you the hugging face transformers library.",
     "It's so doomed!"])

# for result in results:
#     print(result)

tokens = tokenizer.tokenize(
    "We are very happy to show you the hugging face transformers library.")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
token_ids2 = tokenizer(
    "We are very happy to show you the hugging face transformers library.")

# ['we', 'are', 'very', 'happy', 'to', 'show', 'you', 'the', 'hugging', 'face', 'transformers', 'library', '.']
# print(tokens)

# [2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 17662, 2227, 19081, 3075, 1012]
# print(token_ids)

# [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 17662, 2227, 19081, 3075, 1012, 102]
# print(token_ids2['input_ids'])  # 101 and 102 are beginning and ending vals


X_train = ["We are very happy to show you the hugging face transformers library.",
           "I hate it here!"]
batch = tokenizer(X_train, padding=True, truncation=True,
                  max_length=512, return_tensors='pt')

with torch.no_grad():
    outputs = model(**batch, labels=torch.tensor([1, 0]))
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)


save_dir = "saved"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForSequenceClassification.from_pretrained(save_dir)
