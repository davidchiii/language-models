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


# create pipeline
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")
pl = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
              tokenizer=tokenizer, framework='pt')

# sentiment analysis
input = st.text_area('Enter a phrase and press enter to analyze it:',
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


# |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
# |          fine tune          |
# |_____________________________|


# def read_file(dir):
#     dir = Path(dir)
#     exp = []
#     labels = []
#     for label_dir in []
