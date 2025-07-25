# 🌿 AgroShield: Cotton Leaf Disease Detection System

[![Hugging Face Spaces](https://img.shields.io/badge/Deploy%20on-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/david-0705/AgroShield)

## 🔬 Overview

**AgroShield** is an AI-powered web application designed to detect and analyze diseases in cotton leaves using image processing and deep learning. It offers farmers and researchers a reliable tool to:
- Predict diseases from cotton leaf images
- Measure severity using pixel analysis
- Recommend treatment plans
- Maintain a prediction history

🚀 Live Demo:  
👉 [AgroShield on Hugging Face Spaces](https://huggingface.co/spaces/david-0705/AgroShield)

---

## 🧠 Key Features

- **Deep Learning Prediction:** Classifies leaf diseases using a trained TensorFlow model.
- **Severity Analysis:** Uses OpenCV to analyze the infected area percentage.
- **Treatment Recommendation:** Provides actionable remedies for common cotton diseases.
- **Image Upload & Camera Capture:** Accepts files or uses webcam for input.
- **Prediction History:** Saves past predictions and images locally in a CSV log.
- **Streamlit UI:** Clean and interactive interface for real-time feedback.

---

## 📁 File Structure
      AgroShield/
      ├── streamlit_app.py # Main Streamlit app
      ├── model/
      │ └── trained_model5.keras # Trained Keras model for classification
      ├── images/ # Uploaded image cache
      ├── prediction_history.csv # Stores prediction logs
      ├── requirements.txt # Python dependencies
      └── README.md # Project documentation (this file)



---

## 🖼️ Disease Classes Detected

The model currently supports classification of the following:

1. **Bacterial Blight**
2. **Curl Virus**
3. **Fusarium Wilt**
4. **Healthy**

---

## ⚙️ How It Works

1. **Image Input:** Upload or capture an image of a cotton leaf.
2. **Preprocessing:** Image is resized, normalized, and prepared for prediction.
3. **Model Inference:** The TensorFlow model returns the most likely disease class.
4. **Severity Detection:** Infected areas are segmented and quantified.
5. **Output:** Displays disease class, confidence score, severity %, and treatments.
6. **Logging:** Records image and prediction details for future reference.

---

## 🧾 Dependencies

```txt
streamlit
tensorflow
opencv-python-headless
pandas
pillow

