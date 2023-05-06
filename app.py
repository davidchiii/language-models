# |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
# |          import             |
# |_____________________________|

# import the required libraries
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

# |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
# |          models             |
# |_____________________________|
# initial model is defined here. Initial model is the distilbert base finetuned sstv2 english model
# using the transformers library, its easy to just call the package and it works
# technically not even needed since I reinstantiate these values after choosing the model
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
# ui elements on the page
# local media elements are stored in the /media folder.
st.markdown("# :red[Toxic] or Not.")
st.caption("An implementation of a tweet language analyzer.")
# st.divider()
# image
image = Image.open('media/L_two.png')
st.image(image)
# st.divider()

# |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|
# |       choose the model      |
# |_____________________________|
# option stores the string value of which model is chosen
# by default, 'distilbert-base-uncased-finetuned-sst-2-english' is chosen.
option = st.selectbox(
    'What model would you like to use?',
    ('distilbert-base-uncased-finetuned-sst-2-english', 'fine-trained-distilbert')
)


# if we choose the first model
if option == 'distilbert-base-uncased-finetuned-sst-2-english':
    # create pipeline
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english")
    pl = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
                  tokenizer=tokenizer, framework='pt')
    # sentiment analysis

    # Text above the text box
    # default value
    input = st.text_area('Enter a phrase and press ctrl-enter to analyze it:',
                         'grrrr jappan 🇯🇵 is best country in teh world (sekai) !!!!🤬😡!!!👹🤬!!!!! west bAd grrrgghhhg japenis culture⛩🎎🎏 better than amrican🗽🍔👎!!! (>~<) vendor machine eveywhere 🗼and sakura trees are so 🌸 a e s t h e t i c 🌸 UwU if u hate it then your NOT a man of culture so shinē!!! ~hmph baka -_- 🏮')
    result = pl(input)
    # st.json(result)

    # more ui elements to print out the score as well as whether it is NEGATIVE or POSITIVE
    if result[0]["label"] == "NEGATIVE":
        st.markdown(emoji.emojize("Text entry is negative :thumbsdown:"))
        st.write(result[0]["score"])
        st.warning("negative score :<", icon="⚠️")

    elif result[0]["label"] == "POSITIVE":
        st.markdown(emoji.emojize("Text entry is positive :thumbsup:"))
        st.write(result[0]["score"])
        st.success("positive score!")
        st.balloons()
    else:
        st.markdown(emoji.emojize("something went wrong :x:"))

# apparently, switch/case doesn't work in the python parser on huggingface
# therefore, we use elif
elif option == 'fine-trained-distilbert':
    # initialze values for API call
    # query the api because i couldn't find a way to get the secrets to work
    # please don't steal my api key q-q
    def query(payload):
        headers = {
            "Authorization": f"Bearer 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"
        }
        API_URL = "https://api-inference.huggingface.co/models/davidchiii/fine-trained-distilbert"
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    # 🤨

    # input area
    input = st.text_area('Enter a phrase and press ctrl-enter to analyze it:',
                         'grrrr jappan 🇯🇵 is best country in teh world (sekai) !!!!🤬😡!!!👹🤬!!!!! west bAd grrrgghhhg japenis culture⛩🎎🎏 better than amrican🗽🍔👎!!! (>~<) vendor machine eveywhere 🗼and sakura trees are so 🌸 a e s t h e t i c 🌸 UwU if u hate it then your NOT a man of culture so shinē!!! ~hmph baka -_- 🏮')

    # call query to request for the API to run the model
    output = query({
        "inputs": input,
    })

    # st.write(type(output[0]))
    # output the labels
    if not output:
        pass
    # write values to a dictionary to sort them
    dict = {}
    label_cols = ['toxic', 'severe_toxic', 'obscene',
                  'threat', 'insult', 'identity_hate']

    # st.write(output[0][0]['label'])
    # st.write(output[0][1]['label'])
    # st.write(output[0][2]['label'])
    # st.write(output[0][3]['label'])
    # st.write(output[0][4]['label'])
    # st.write(output[0][5]['label'])
    # st.write(output)

    # set values for local dictionary
    for i, key in enumerate(output[0]):
        if key['label'] == "LABEL_0":
            dict["toxic"] = key["score"]
        elif key['label'] == "LABEL_1":
            dict["severe_toxic"] = key["score"]
        elif key['label'] == "LABEL_2":
            dict["obscene"] = key["score"]
        elif key['label'] == "LABEL_3":
            dict["threat"] = key["score"]
        elif key['label'] == "LABEL_4":
            dict["insult"] = key["score"]
        elif key['label'] == "LABEL_5":
            dict["identity_hate"] = key["score"]

    # read through all dictionary values to determine the largest
    largest = 'toxic'
    for key in dict.keys():
        if dict[largest] < dict[key]:
            largest = key

    # create and write the front end
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Text:")
        st.write(input)
    with col2:
        st.subheader("Most Prevalent Label:")
        # should be toxic for most of the time, but it's vague
        st.write(largest)
    with col3:
        st.subheader("Value:")
        st.write(dict)
