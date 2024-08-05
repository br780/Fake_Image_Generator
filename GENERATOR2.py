#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import sys


# In[2]:


st.title('IMAGE ART GENERATOR: AFRICAN FASHION EDITION')


# Add the StyleGAN3 directory to the Python path
stylegan3_path = "C:/Users/DELL/Desktop/Jupyter files/New folder/bryan_ai/stylegan3"
sys.path.append(stylegan3_path)

import dnnlib
import legacy

def load_model(model_path):
    with dnnlib.util.open_url(model_path) as f:
        model = legacy.load_network_pkl(f)['G_ema']
    model.eval()
    return model
# Load your StyleGAN3 model
#def load_model(model_path):
 #   model = torch.load(model_path, map_location=torch.device('cpu'))
  #  model.eval()
  #  return model

# Function to generate an image using StyleGAN3
def generate_image(model, z):
    with torch.no_grad():
        img = model(z)
    img = (img.clamp(-1, 1) + 1) / 2.0 * 255
    img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return img[0]

# Load the model
model_path = "C:/Users/DELL/Desktop/Jupyter files/New folder/bryan_ai/stylegan3/model/00009-stylegan3-t-new_clothing-gpus1-batch20-gamma8.2/network-snapshot-000400.pkl"
model = load_model(model_path)

# Streamlit interface
st.title("StyleGAN3 Image Generator and Classifier")

# Input fields for generating images
st.sidebar.header("Image Generation")
seed = st.sidebar.number_input("Seed", value=0, min_value=0, max_value=100000)
z_dim = 512  # Dimension of the latent vector

# Generate button
if st.sidebar.button("Generate Image"):
    torch.manual_seed(seed)
    z = torch.randn(1, z_dim)
    generated_img = generate_image(model, z)
    st.image(generated_img, caption="Generated Image", use_column_width=True)

# Input field for classification
st.sidebar.header("Image Classification")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png"])

# Function to preprocess the uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Classification button
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)

    if st.sidebar.button("Classify Image"):
        image = preprocess_image(image)
        with torch.no_grad():
            classification = model(image)
        st.write(f"Classification Result: {classification.argmax(1).item()}")

st.markdown("""
<style>
    .main {
        font-family: 'Orbitron', sans-serif;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        width: 100%;
        padding: 10px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)


# In[ ]:





# In[ ]:




