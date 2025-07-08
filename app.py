import os
import streamlit as st
import numpy as np
from PIL import Image
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Automatically download model from Google Drive if not present
model_path = "cnn_model.h5"
if not os.path.exists(model_path):
    file_id = "1ROM98O3v5qln-3dYzPVgp0dL6_LCjq8y"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

@st.cache_resource
def load_my_model():
    return load_model("cnn_model.h5")

model = load_my_model()
class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Page config
st.set_page_config(page_title="Alzheimer's MRI Classifier", layout="wide")

# Header
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        padding: 10px;
    }
    .sub-header {
        font-size: 20px;
        color: #34495e;
        text-align: center;
        margin-bottom: 30px;
    }
    .disease-title {
        font-size: 24px;
        font-weight: 600;
        color: #2c3e50;
    }
    .symptom-list {
        font-size: 16px;
        color: #2f3640;
    }
</style>
<div class="main-title">🧠 Alzheimer's Disease Detection from MRI</div>
<div class="sub-header">Upload a brain MRI scan to detect Alzheimer's stage using deep learning</div>
""", unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("L")
        image = image.resize((176, 176))
        img_array = keras_image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        st.image(image, caption="🖼 Uploaded MRI (Grayscale)", use_column_width=False, width=300)

        with st.spinner("🧠 Analyzing MRI..."):
            prediction = model.predict(img_array)[0]
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = prediction[predicted_index] * 100

        st.success(f"🔍 *Prediction*: {predicted_class}")
        st.info(f"📈 *Confidence*: {confidence:.2f}%")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.subheader("📘 Alzheimer's Disease Stages")

col1, col2 = st.columns(2)

with col1:
    with st.expander("🟢 Non Demented"):
        st.markdown("""
        <div class="disease-title">Description:</div>
        <ul class="symptom-list">
            <li>Healthy cognitive state with no symptoms</li>
            <li>Baseline for comparison in diagnosis</li>
            <li>Independent daily functioning</li>
        </ul>
        <div class="disease-title">Recommended Action:</div>
        <ul class="symptom-list">
            <li>Maintain brain health with physical & mental activity</li>
            <li>Regular medical checkups</li>
            <li>Nutritious diet and low stress lifestyle</li>
        </ul>
        <a href="https://www.alz.org/alzheimers-dementia/stages" target="_blank">🔗 Learn more about Alzheimer's stages</a>
        """, unsafe_allow_html=True)

    with st.expander("🟡 Very Mild Demented"):
        st.markdown("""
        <div class="disease-title">Description:</div>
        <ul class="symptom-list">
            <li>Minor memory lapses (not affecting daily life)</li>
            <li>Often considered part of normal aging</li>
            <li>Detected only through testing</li>
        </ul>
        <div class="disease-title">Recommended Action:</div>
        <ul class="symptom-list">
            <li>Mental stimulation (reading, puzzles, learning)</li>
            <li>Frequent social interaction</li>
            <li>Monitor for any increasing symptoms</li>
        </ul>
        <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6527027/" target="_blank">🔗 Read scientific article on early Alzheimer's</a>
        """, unsafe_allow_html=True)

with col2:
    with st.expander("🟠 Mild Demented"):
        st.markdown("""
        <div class="disease-title">Description:</div>
        <ul class="symptom-list">
            <li>Memory and thinking issues noticeable to others</li>
            <li>Problems with planning and completing tasks</li>
            <li>Difficulty remembering names and recent events</li>
        </ul>
        <div class="disease-title">Recommended Action:</div>
        <ul class="symptom-list">
            <li>Consult a neurologist for evaluation</li>
            <li>Establish a treatment or care routine</li>
            <li>Start safety planning at home</li>
        </ul>
        <a href="https://www.nia.nih.gov/health/what-mild-cognitive-impairment" target="_blank">🔗 What is Mild Cognitive Impairment?</a>
        """, unsafe_allow_html=True)

    with st.expander("🔴 Moderate Demented"):
        st.markdown("""
        <div class="disease-title">Description:</div>
        <ul class="symptom-list">
            <li>Significant confusion and forgetfulness</li>
            <li>Help needed for dressing, eating, etc.</li>
            <li>Personality and behavior changes possible</li>
        </ul>
        <div class="disease-title">Recommended Action:</div>
        <ul class="symptom-list">
            <li>24/7 care and support structure is essential</li>
            <li>Medication and therapies may slow progression</li>
            <li>Family and caregiver education crucial</li>
        </ul>
        <a href="https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/in-depth/alzheimers-stages/art-20048448" target="_blank">🔗 Moderate Alzheimer's: Mayo Clinic</a>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2025 | Alzheimer's Detection System | Powered by Deep Learning")
   
            
        
          
