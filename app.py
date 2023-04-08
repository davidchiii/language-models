import streamlit as st
from transformers import pipeline, AutoModel, AutoTokenizer
# from transformers import Autotokenizer
# import json
import torch

from PIL import Image

# import models
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
bertweet = AutoModel.from_pretrained("vinai/bertweet-base")



# title
st.title("Toxic or Not.")
st.caption("An implementation of a language analyzer.")
# st.divider() doesnt work????
# image
image = Image.open('media/L_two.png')
st.image(image)



# test
input = 'I love using this library to implement this function!'

input_ids = torch.tensor([tokenizer.encode(input)])
with torch.no_grad():
    results = bertweet(input_ids)
st.write("Results: 'I love using this library to implement this function!'")
# for i in results:
#     st.write(i['label'])
#     st.write("Score:",i['score'])
st.write(results)



# sentiment analysis
input = st.text_input('Enter a phrase:', 'I hate anime.')
input_ids = torch.tensor([tokenizer.encode(input)])
with torch.no_grad():
    results = bertweet(input_ids)
st.write("analyis", results)



# file_name = st.file_uploader("Upload a hot dog candidate image")

# if file_name is not None:
#     col1, col2 = st.columns(2)

#     image = Image.open(file_name)
#     col1.image(image, use_column_width=True)
#     predictions = pipeline(image)

#     col2.header("Probabilities")
#     for p in predictions:
#         col2.subheader(f"{ p['label'] }: { round(p['score'] * 100, 1)}%")