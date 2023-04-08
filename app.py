import streamlit as st
from transformers import pipeline
# from transformers import Autotokenizer

# from PIL import Image


pipeline = pipeline(task="sentiment-analysis")
results = pipeline("I love using this library to implement this function!")

# title
st.title("Toxic or Not.")
st.caption("An implementation of a language analyzer.")
# st.divider()


# test
st.write("Results: 'I love using this library to implement this function!'")
st.write(results)

results = st.text_input('Enter a phrase:', 'I hate anime.')
st.write("analyis", pipeline(results))





# file_name = st.file_uploader("Upload a hot dog candidate image")

# if file_name is not None:
#     col1, col2 = st.columns(2)

#     image = Image.open(file_name)
#     col1.image(image, use_column_width=True)
#     predictions = pipeline(image)

#     col2.header("Probabilities")
#     for p in predictions:
#         col2.subheader(f"{ p['label'] }: { round(p['score'] * 100, 1)}%")