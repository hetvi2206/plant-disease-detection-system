# 🌿 Plant Disease Detection System for Sustainable Agriculture

![Plant Health](Diseases.png)

## 📌 Overview

This project is a **Plant Disease Detection System** built using **Streamlit** and **TensorFlow**, designed to assist farmers and agricultural experts in identifying plant diseases through image recognition. It promotes sustainable farming by enabling early diagnosis and treatment of plant health issues.

---

## 🚀 Features

- 📸 Upload plant leaf images to detect possible diseases.
- 🧠 Deep learning model trained on plant disease datasets.
- 🎨 Clean, user-friendly **Streamlit** interface.
- 🌱 Supports multiple plant types and disease classifications.
- 🎉 Visual effects (like snow animation) to enhance UX on predictions.

---

## 🛠️ Technologies Used

- **Python 3.12+**
- **TensorFlow / Keras**
- **Streamlit**
- **NumPy**
- **PIL**
- **gdown** (for model download during deployment)

---

## 📦 Installation & Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/plant-disease-detection.git
   cd plant-disease-detection

2. **pip install -r requirements.txt**
   ```bash
   pip install -r requirements.txt

3. **Run the Streamlit app**
   ```bash
   streamlit run main.py

---

## 📊 Model Training

- The model was trained using a CNN (Convolutional Neural Network) architecture.
- Images resized to 128x128
- Uses a multi-class classification approach for detecting plant diseases.

---

## 📸 Sample Prediction

- Upload an image
-  Click Predict
- The system displays the predicted disease class with a snow effect 🎉
