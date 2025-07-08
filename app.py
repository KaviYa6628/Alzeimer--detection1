import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load model and class names
model = load_model("cnn_model.h5")
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
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Preprocess image to match model input
        image = Image.open(uploaded_file).convert("L")
        image = image.resize((176, 176))
        img_array = keras_image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        st.image(image, caption="🖼 Uploaded MRI (Grayscale)", use_column_width=False, width=300)

        # Predict
        with st.spinner("Analyzing MRI with AI model..."):
            prediction = model.predict(img_array)[0]
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = prediction[predicted_index] * 100

        st.success(f"🔍 **Prediction**: {predicted_class}")
        st.info(f"📈 **Confidence**: {confidence:.2f}%")

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

st.markdown("---")
st.subheader("🧬 Understanding Alzheimer's Disease Stages")

col1, col2 = st.columns(2)

with col1:
    with st.expander("🟢 Non Demented"):
        st.markdown("""
        <div class="disease-title">What it means:</div>
        <ul class="symptom-list">
            <li>Healthy brain with no cognitive impairment</li>
            <li>No signs of memory loss or confusion</li>
            <li>Baseline control group in diagnosis</li>
        </ul>
        <div class="disease-title">Recommended Action:</div>
        <p>Maintain a healthy lifestyle, regular exercise, and balanced diet.</p>
        """, unsafe_allow_html=True)

    with st.expander("🟡 Very Mild Demented"):
        st.markdown("""
        <div class="disease-title">What it means:</div>
        <ul class="symptom-list">
            <li>Minor forgetfulness not affecting daily tasks</li>
            <li>Normal aging-related memory loss</li>
            <li>Often unrecognized without testing</li>
        </ul>
        <div class="disease-title">Symptoms:</div>
        <ul class="symptom-list">
            <li>Forgetting names or appointments</li>
            <li>Misplacing objects occasionally</li>
        </ul>
        <div class="disease-title">Recommended Action:</div>
        <p>Monitor regularly; engage in mental activities, stay social.</p>
        """, unsafe_allow_html=True)

with col2:
    with st.expander("🟠 Mild Demented"):
        st.markdown("""
        <div class="disease-title">What it means:</div>
        <ul class="symptom-list">
            <li>Noticeable cognitive decline</li>
            <li>Interference with work and social life</li>
        </ul>
        <div class="disease-title">Symptoms:</div>
        <ul class="symptom-list">
            <li>Getting lost in familiar places</li>
            <li>Difficulty remembering recent events</li>
            <li>Problems with planning or organization</li>
        </ul>
        <div class="disease-title">Recommended Action:</div>
        <p>Seek medical diagnosis, begin therapy or treatment early.</p>
        """, unsafe_allow_html=True)

    with st.expander("🔴 Moderate Demented"):
        st.markdown("""
        <div class="disease-title">What it means:</div>
        <ul class="symptom-list">
            <li>Worsening memory and confusion</li>
            <li>Assistance needed for daily tasks</li>
        </ul>
        <div class="disease-title">Symptoms:</div>
        <ul class="symptom-list">
            <li>Inability to recognize family members</li>
            <li>Loss of orientation to time/place</li>
            <li>Repeating questions or statements</li>
        </ul>
        <div class="disease-title">Recommended Action:</div>
        <p>Medical intervention required; consider support for caregiving and daily living assistance.</p>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed with 🧠 by AI-Powered Diagnostics | Data sourced from Alzheimer's Association & Mayo Clinic")
