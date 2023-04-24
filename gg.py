
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

import requests

API_URL = "https://api-inference.huggingface.co/models/davidchiii/fine-trained-distilbert"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": "I like you. I love you",
})

print(output)
