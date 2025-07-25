import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import uuid
import cv2

# Constants
PRED_HISTORY_FILE = "prediction_history.csv"
IMAGE_DIR = "images"

# Ensure the image directory exists
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Cotton Disease Recommender Class
class CottonDiseaseRecommender:
    def __init__(self):
        # Knowledge base mapping diseases to treatments
        self.treatment_database = {
            "bacterial_blight": {
                "fertilizers": ["Copper-based fertilizers", "Calcium nitrate"],
                "nutrients": ["Copper", "Zinc", "Boron"],
                "pesticides": ["Copper oxychloride", "Streptomycin sulfate"],
                "cultural_practices": ["Remove infected plants", "Crop rotation", "Use disease-free seeds"]
            },
            "curl_virus": {
                "fertilizers": ["Balanced NPK fertilizer", "Micronutrient mix"],
                "nutrients": ["Potassium", "Calcium", "Magnesium"],
                "pesticides": ["Imidacloprid", "Thiamethoxam", "Neem oil"],
                "cultural_practices": ["Remove infected plants", "Control whitefly vectors", "Use virus-resistant varieties"]
            },
            "fussarium_wilt": {
                "fertilizers": ["Phosphorus-rich fertilizers", "Trichoderma-enriched fertilizers"],
                "nutrients": ["Calcium", "Phosphorus", "Silicon"],
                "pesticides": ["Fungicides with thiophanate-methyl", "Carbendazim", "Benomyl"],
                "cultural_practices": ["Plant resistant varieties", "Soil solarization", "Long crop rotations"]
            },
            "healthy": {
                "fertilizers": ["Balanced NPK fertilizer", "Organic compost"],
                "nutrients": ["Complete micronutrient mix", "Nitrogen", "Phosphorus", "Potassium"],
                "pesticides": ["Preventative application of neem oil"],
                "cultural_practices": ["Regular irrigation", "Proper spacing", "Weed control"]
            }
        }
        
        # Severity levels and corresponding treatment adjustments
        self.severity_adjustments = {
            "low": {
                "message": "Early stage detection. Preventative treatments recommended.",
                "fertilizer_dose": "Standard dose",
                "pesticide_dose": "Minimal application"
            },
            "medium": {
                "message": "Moderate infection detected. Prompt treatment required.",
                "fertilizer_dose": "Standard dose with foliar spray",
                "pesticide_dose": "Standard application"
            },
            "high": {
                "message": "Severe infection detected. Aggressive treatment required.",
                "fertilizer_dose": "Increased dose with repeated applications",
                "pesticide_dose": "Maximum recommended application"
            }
        }
    
    def get_severity_level(self, severity_percentage):
        """Determine severity level based on percentage"""
        if severity_percentage < 20:
            return "low"
        elif severity_percentage < 50:
            return "medium"
        else:
            return "high"
    
    def get_recommendations(self, disease_name, severity_percentage):
        """
        Get treatment recommendations for a specific cotton disease
        
        Args:
            disease_name (str): The detected disease name
            severity_percentage (float): Severity percentage from image analysis
            
        Returns:
            dict: Treatment recommendations
        """
        if disease_name not in self.treatment_database:
            return {
                "status": "error",
                "message": f"Unknown disease: {disease_name}. Please update the database."
            }
        
        # Determine severity level
        severity = self.get_severity_level(severity_percentage)
        
        # Get base recommendations for the disease
        recommendations = self.treatment_database[disease_name].copy()
        
        # Add severity-specific adjustments
        if severity in self.severity_adjustments:
            recommendations["severity"] = severity
            recommendations["severity_guidance"] = self.severity_adjustments[severity]
        
        return {
            "status": "success",
            "disease": disease_name,
            "severity": severity,
            "recommendations": recommendations
        }

# Load prediction history from CSV
def load_prediction_history():
    if os.path.exists(PRED_HISTORY_FILE):
        return pd.read_csv(PRED_HISTORY_FILE)
    else:
        return pd.DataFrame(columns=["prediction", "image_path", "recommendations"])

# Save prediction history to CSV
def save_prediction_history(prediction, image_file, recommendations=None):
    history = load_prediction_history()
    # Use a unique filename by combining UUID with the original filename
    unique_image_name = f"{uuid.uuid4()}_{image_file}"
    image_path = os.path.join(IMAGE_DIR, unique_image_name)
    
    # Save the image if it doesn't already exist
    with open(image_path, "wb") as f:
        f.write(test_image.getbuffer())
    
    new_entry = pd.DataFrame({
        "prediction": [prediction], 
        "image_path": [image_path],
        "recommendations": [str(recommendations) if recommendations else ""]
    })
    history = pd.concat([history, new_entry], ignore_index=True)
    history.to_csv(PRED_HISTORY_FILE, index=False)

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = Image.open(test_image)
    image = image.convert("RGB")  # Ensure the image is in RGB format
    image = image.resize((256, 256))  # Resize to the input size expected by the model
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return predictions[0]  # Return the prediction probabilities for each class

# Severity Calculation
def calculate_severity(test_image):
    img = Image.open(test_image)
    img = img.convert("RGB")
    img = np.array(img)
    
    # Convert to grayscale for total leaf area (TLA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, leaf_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Convert to HSV for infected leaf area (ILA)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_bound = np.array([10, 40, 40])  # Customize these ranges based on dataset
    upper_bound = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Calculate Total Leaf Area (TLA) and Infected Leaf Area (ILA)
    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    infected_pixels = cv2.countNonZero(mask)
    
    # Calculate Severity
    if total_leaf_pixels > 0:
        severity_percentage = (infected_pixels / total_leaf_pixels) * 100
    else:
        severity_percentage = 0
    
    return severity_percentage, total_leaf_pixels, infected_pixels

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Prediction History"])

# Initialize prediction history in session state if it doesn't exist
if 'pred_history' not in st.session_state:
    st.session_state.pred_history = []  # To store tuples of (prediction, image)

# Main Page
if app_mode == "Home":
    st.header("COTTON LEAF DISEASE RECOGNITION SYSTEM")
    image_path = "plant2.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown(""" 
    Welcome to the Cotton Leaf Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying cotton plant diseases efficiently and provide treatment recommendations. Upload an image of a cotton plant leaf, and our system will analyze it to detect any signs of diseases and recommend appropriate fertilizers, nutrients, and pesticides. Together, let's protect our cotton crops and ensure a healthier harvest!
    
    ### How It Works
    1. *Upload Image:* Go to the *Disease Recognition* page and upload an image of a cotton plant leaf with suspected diseases.
    2. *Analysis:* Our system will process the image using advanced algorithms to identify potential diseases and their severity.
    3. *Results:* View the results and get recommendations for fertilizers, nutrients, and pesticides to treat the identified disease.
    
    ### Why Choose Us?
    - *Accuracy:* Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - *Comprehensive:* Not just detection, but also treatment recommendations based on disease and severity.
    - *User-Friendly:* Simple and intuitive interface for seamless user experience.
    - *Fast and Efficient:* Receive results in seconds, allowing for quick decision-making.
    
    ### Get Started
    Click on the *Disease Recognition* page in the sidebar to upload an image and experience the power of our Cotton Disease Recognition System!
    
    ### About Us
    Learn more about the project, our team, and our goals on the *About* page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown(""" 
        #### About Dataset
        This dataset consists of images of healthy and diseased cotton crop leaves categorized into 4 different classes:
        - Bacterial Blight
        - Curl Virus
        - Fussarium Wilt
        - Healthy
        
        #### About Our Approach
        Our system uses a deep learning model to classify cotton leaf diseases and estimate their severity. Beyond just detection, we provide personalized treatment recommendations including:
        
        - Suitable fertilizers
        - Required nutrients
        - Appropriate pesticides
        - Best cultural practices
        
        The recommendations are tailored based on both the disease identified and its severity level, ensuring that farmers can take the most effective actions to protect their crops.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    uploaded_image = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
    camera_image = st.camera_input("Or take a photo with your camera:")

    # Use the camera image if available
    if camera_image:
        test_image = camera_image
    elif uploaded_image is not None:
        test_image = uploaded_image
    else:
        test_image = None

    if test_image is not None:
        # Show the selected image
        st.image(test_image, caption="Selected Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        if test_image is not None:
            # st.leaf()  # Fun snow effect
            st.write("Our Prediction")

            # Perform model prediction
            predictions = model_prediction(test_image)
            
            # Reading Labels
            class_name = ['bacterial_blight', 'curl_virus', 'fussarium_wilt', 'healthy']
            bacterial_blight = predictions[0] * 100
            curl_virus = predictions[1] * 100
            fussarium_wilt = predictions[2] * 100
            healthy = predictions[3] * 100
            
            # Get the predicted disease
            predicted_disease = class_name[np.argmax(predictions)]
            
            # Display disease prediction results
            st.success(f"Prediction: {predicted_disease}")
            st.write(f"bacterial blight: {bacterial_blight:.2f}%")
            st.write(f"curl virus: {curl_virus:.2f}%")
            st.write(f"fussarium wilt: {fussarium_wilt:.2f}%")
            st.write(f"healthy: {healthy:.2f}%")

            # Calculate severity
            severity_percentage, total_leaf_pixels, infected_pixels = calculate_severity(test_image)

            # Show the calculation steps
            st.write(f"**Total Leaf Area (TLA)**: {total_leaf_pixels} pixels")
            st.write(f"**Infected Leaf Area (ILA)**: {infected_pixels} pixels")
            st.write(f"**Severity Calculation**: ({infected_pixels} / {total_leaf_pixels}) × 100 = {severity_percentage:.2f}%")
            
            # Display the severity result
            st.write(f"**Severity**: {severity_percentage:.2f}%")
            
            # Get treatment recommendations
            recommender = CottonDiseaseRecommender()
            recommendations = recommender.get_recommendations(predicted_disease, severity_percentage)
            
            # Display treatment recommendations
            st.header("Treatment Recommendations")
            
            # Determine severity level
            severity_level = recommendations["severity"]
            st.write(f"**Severity Level**: {severity_level.capitalize()}")
            st.write(f"**Guidance**: {recommendations['recommendations']['severity_guidance']['message']}")
            
            # Display fertilizer recommendations
            st.subheader("Recommended Fertilizers")
            st.write(f"**Dosage**: {recommendations['recommendations']['severity_guidance']['fertilizer_dose']}")
            for fertilizer in recommendations['recommendations']['fertilizers']:
                st.write(f"- {fertilizer}")
            
            # Display nutrient recommendations
            st.subheader("Recommended Nutrients")
            for nutrient in recommendations['recommendations']['nutrients']:
                st.write(f"- {nutrient}")
            
            # Display pesticide recommendations
            st.subheader("Recommended Pesticides")
            st.write(f"**Dosage**: {recommendations['recommendations']['severity_guidance']['pesticide_dose']}")
            for pesticide in recommendations['recommendations']['pesticides']:
                st.write(f"- {pesticide}")
            
            # Display cultural practices
            st.subheader("Cultural Practices")
            for practice in recommendations['recommendations']['cultural_practices']:
                st.write(f"- {practice}")
            
            # Save prediction history with recommendations
            prediction_summary = f"Disease: {predicted_disease}, Severity: {severity_percentage:.2f}%"
            save_prediction_history(prediction_summary, test_image.name, recommendations)
        else:
            st.error("Please upload an image or take a photo before predicting.")

# Prediction History Page
elif app_mode == "Prediction History":
    st.header("Prediction History")
    
    # Load prediction history
    prediction_history = load_prediction_history()
    
    # Show prediction history
    if not prediction_history.empty:
        for idx, row in prediction_history.iterrows():
            st.subheader(f"Prediction {idx + 1}")
            st.write(f"**Result**: {row['prediction']}")
            
            # Display recommendations if available
            if row['recommendations'] and row['recommendations'] != "":
                st.write("**Recommendations were provided**")
                st.write("(View details in the Disease Recognition page)")
            
            # Display image
            image_path = row['image_path']
            # Check if the image file exists before displaying
            if os.path.isfile(image_path):
                st.image(image_path, caption=f"Image {idx + 1}", use_column_width=True)
            else:
                st.error(f"Image file '{image_path}' not found.")
            
            st.markdown("---")
    else:
        st.write("No predictions made yet.")