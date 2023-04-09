import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, DistilBertForSequenceClassification
# from transformers import Autotokenizer
# import json
import torch
import emoji
from PIL import Image


# import models
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
pl = pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english", tokenizer=tokenizer, framework='pt')
# title
st.markdown("# :red[Toxic] or Not.")
st.caption("An implementation of a tweet language analyzer.")
# st.divider()
# image
image = Image.open('media/L_two.png')
st.image(image)
# st.divider()



# create pipeline
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
pl = pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english", tokenizer=tokenizer, framework='pt')

# sentiment analysis
input = st.text_input('Enter a phrase and press enter to analyze it:', 'I hate anime.')
result = pl(input)
# st.json(result)

if result[0]["label"] == "NEGATIVE":
    st.markdown(emoji.emojize("Text entry is negative :thumbsdown:"))
    st.write(result[0]["score"])
elif result[0]["label"] == "POSITIVE":
    st.markdown(emoji.emojize("Text entry is positive :thumbsup:"))
    st.write(result[0]["score"])
else:
    st.markdown(emoji.emojize("something went wrong :x:"))
# file_name = st.file_uploader("Upload a hot dog candidate image")

# if file_name is not None:
#     col1, col2 = st.columns(2)

#     image = Image.open(file_name)
#     col1.image(image, use_column_width=True)
#     predictions = pipeline(image)

#     col2.header("Probabilities")
#     for p in predictions:
#         col2.subheader(f"{ p['label'] }: { round(p['score'] * 100, 1)}%")