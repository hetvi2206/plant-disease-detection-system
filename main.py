import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Custom CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
            border-right: 2px solid #e0e0e0;
        }
        h1 {
            font-family: 'Arial', sans-serif;
            font-weight: bold;
            color: #2e7d32;
        }
        h2, h3, h4, h5 {
            font-family: 'Arial', sans-serif;
            color: #4caf50;
        }
        button {
            background-color: #4caf50 !important;
            color: white !important;
            border-radius: 5px !important;
            padding: 10px 20px !important;
        }
        .stButton button:hover {
            background-color: #357a38 !important;
        }
        .uploaded-file-info {
            font-family: 'Courier New', Courier, monospace;
            font-size: 14px;
        }
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# Download Model from Google Drive if not present
model_path = "trained_plant_disease_model.keras"
drive_file_id = "1TO-iQJrnVvIS7b6xONn7Lr__R59WxWZi"
gdrive_url = f"https://drive.google.com/uc?id={drive_file_id}"
if not os.path.exists(model_path):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(gdrive_url, model_path, quiet=False)

# Model Prediction Function
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Import Image for Main Page
img = Image.open("Diseases.png")

# Main Page Logic
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    st.image(img, caption="Understanding Plant Health", use_column_width=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("📸 Plant Disease Recognition")
    test_image = st.file_uploader("Upload an Image:", type=["jpg", "png", "jpeg"])
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.snow()
        result_index = model_prediction(test_image)
        class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
            'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
            'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        st.success(f"The model predicts: **{class_names[result_index]}**")
