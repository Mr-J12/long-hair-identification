import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os

# --- Configuration ---
MODELS_DIR = 'models'
IMG_SIZE = 128

# Set page config
st.set_page_config(page_title="Long Hair Identifier", layout="wide")

# --- Model Loading ---
@st.cache_resource
def load_all_models():
    """Load all three trained models."""
    try:
        age_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'age_model.keras'))
        gender_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'gender_model.keras'))
        hair_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'hair_model.keras'))
        return age_model, gender_model, hair_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure you have trained the models by running `python train.py --task <task_name>` for all three tasks (age, gender, hair).")
        return None, None, None

# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocesses the image for model prediction."""
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --- Final Logic Function ---
def get_final_prediction(age, gender, hair):
    """Applies the project's core logic and returns all results."""
    predicted_age = int(np.round(age[0][0]))
    gender_prob = gender[0][0]
    hair_prob = hair[0][0]

    display_gender = "Female" if gender_prob > 0.5 else "Male"
    display_hair = "Long" if hair_prob > 0.5 else "Short"

    if 20 <= predicted_age <= 30:
        final_prediction = "Female" if display_hair == "Long" else "Male"
        reason = f"Rule Applied: Age ({predicted_age}) is between 20-30, so prediction is based on hair length."
    else:
        final_prediction = display_gender
        reason = f"Rule Applied: Age ({predicted_age}) is outside 20-30, so prediction is based on predicted biological gender."

    # **FIX 1: Return the probabilities along with the other values**
    return predicted_age, display_gender, display_hair, final_prediction, reason, gender_prob, hair_prob

# --- Streamlit App Layout ---
st.title("🧓 Long Hair Identifier 🧑‍🦰")
st.markdown("This app predicts gender based on a special set of rules involving age and hair length.")

age_model, gender_model, hair_model = load_all_models()

if age_model:
    uploaded_file = st.file_uploader("Choose an image of a person's face...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)

        with col2:
            st.write("### Prediction Results")
            with st.spinner('Analyzing the image...'):
                processed_image = preprocess_image(image)
                pred_age = age_model.predict(processed_image)
                pred_gender = gender_model.predict(processed_image)
                pred_hair = hair_model.predict(processed_image)
                
                p_age, p_gender, p_hair, final_pred, reason, gender_prob, hair_prob = get_final_prediction(pred_age, pred_gender, pred_hair)

                # Display results
                st.info(f"**Final Prediction:** `{final_pred}`")
                st.markdown("---")
                st.write("#### Intermediate Model Outputs:")
                st.write(f"**Predicted Age:** {p_age} years")
                st.write(f"**Predicted Biological Gender:** {p_gender} (Confidence: {gender_prob:.2f})")
                st.write(f"**Predicted Hair Length:** {p_hair} (Confidence: {hair_prob:.2f})")
                st.markdown("---")
                st.write("#### Reasoning:")
                st.warning(reason)
else:
    st.warning("Models are not loaded. Please train them first.")