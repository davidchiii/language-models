# |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
# |          import             |
# |_____________________________|

import streamlit as st
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertForSequenceClassification
import torch
import emoji
import time
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import requests


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


# |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
# |          models             |
# |_____________________________|
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name)  # not used
pl = pipeline("sentiment-analysis", model=model,
              tokenizer=tokenizer, framework='pt')


# |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
# |          title              |
# |_____________________________|
st.markdown("# :red[Toxic] or Not.")
st.caption("An implementation of a tweet language analyzer.")
# st.divider()
# image
image = Image.open('media/L_two.png')
st.image(image)
# st.divider()


option = st.selectbox(
    'What model would you like to use?',
    ('distilbert-base-uncased-finetuned-sst-2-english', 'fine-trained-distilbert')
)

if option == 'distilbert-base-uncased-finetuned-sst-2-english':
    # create pipeline
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english")
    pl = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
                  tokenizer=tokenizer, framework='pt')
    # sentiment analysis
    input = st.text_area('Enter a phrase and press ctrl-enter to analyze it:',
                         'grrrr jappan 🇯🇵 is best country in teh world (sekai) !!!!🤬😡!!!👹🤬!!!!! west bAd grrrgghhhg japenis culture⛩🎎🎏 better than amrican🗽🍔👎!!! (>~<) vendor machine eveywhere 🗼and sakura trees are so 🌸 a e s t h e t i c 🌸 UwU if u hate it then your NOT a man of culture so shinē!!! ~hmph baka -_- 🏮')
    result = pl(input)
    # st.json(result)

    if result[0]["label"] == "NEGATIVE":
        st.markdown(emoji.emojize("Text entry is negative :thumbsdown:"))
        st.write(result[0]["score"])
        st.failure("negative score :<")

    elif result[0]["label"] == "POSITIVE":
        st.markdown(emoji.emojize("Text entry is positive :thumbsup:"))
        st.write(result[0]["score"])
        st.success("positive score!")
    else:
        st.markdown(emoji.emojize("something went wrong :x:"))
elif option == 'fine-trained-distilbert':
    API_URL = "https://api-inference.huggingface.co/models/davidchiii/fine-trained-distilbert"
    headers = {
        "Authorization": "Bearer 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"
    }
    input = st.text_area('Enter a phrase and press ctrl-enter to analyze it:',
                         'grrrr jappan 🇯🇵 is best country in teh world (sekai) !!!!🤬😡!!!👹🤬!!!!! west bAd grrrgghhhg japenis culture⛩🎎🎏 better than amrican🗽🍔👎!!! (>~<) vendor machine eveywhere 🗼and sakura trees are so 🌸 a e s t h e t i c 🌸 UwU if u hate it then your NOT a man of culture so shinē!!! ~hmph baka -_- 🏮')

    output = query({
        "inputs": input,
    })
    dict = {}

    label_cols = ['toxic', 'severe_toxic', 'obscene',
                  'threat', 'insult', 'identity_hate']

    dict[label_cols[0]] = output[0][0]
    dict[label_cols[1]] = output[0][1]
    dict[label_cols[2]] = output[0][2]
    dict[label_cols[3]] = output[0][3]
    dict[label_cols[4]] = output[0][4]
    dict[label_cols[5]] = output[0][5]

    largest = 'toxic'
    for key in dict.keys():
        if dict[largest]['score'] < dict[key]['score']:
            largest = key

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Text:")
        st.write(input)
    with col2:
        st.subheader("Most Prevalent Label:")
        st.write(largest)
    with col3:
        st.subheader("Value:")
        st.write(dict[largest]['score'])
