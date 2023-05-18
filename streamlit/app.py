import streamlit as st
from PIL import Image
import urllib.request
import torch
from torch import nn
import numpy as np
from VT5 import VT5
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    T5Tokenizer, 
    T5ForConditionalGeneration,
    CLIPVisionModelWithProjection,
    AutoProcessor
)

  
clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

vt5 = VT5(t5,tokenizer,clip)
vt5.load_state_dict(torch.load('weights.bin',map_location=torch.device('cpu')))

# Assuming you have this function that generates captions
def generate_caption(image):
    # Your model code here
    caption = "This is a placeholder caption"
  
    caption = vt5.generate_caption(image)
    return caption

st.title("Image Captioning App")
#st.image(image.numpy().reshape(224,224,3), caption='Uploaded Image.', clamp=True,use_column_width=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', clamp=True,use_column_width=True)
    image = processor(images=image,return_tensors='pt').pixel_values
    st.write("")
    st.write("Generating caption...")
    caption = generate_caption(image)
    st.write("Caption: ", caption)
