import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import uuid
import cv2

# Page configuration
st.set_page_config(
    page_title="Cotton Leaf Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Home"
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #2E7D32;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #388E3C;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .severity-low {
        color: #4CAF50;
        font-weight: bold;
        padding: 0.3rem 0.6rem;
        background-color: rgba(76, 175, 80, 0.1);
        border-radius: 1rem;
    }
    .severity-medium {
        color: #FF9800;
        font-weight: bold;
        padding: 0.3rem 0.6rem;
        background-color: rgba(255, 152, 0, 0.1);
        border-radius: 1rem;
    }
    .severity-high {
        color: #F44336;
        font-weight: bold;
        padding: 0.3rem 0.6rem;
        background-color: rgba(244, 67, 54, 0.1);
        border-radius: 1rem;
    }
    .recommend-item {
        padding: 0.5rem;
        margin: 0.3rem 0;
        background-color: #000000;
        border-left: 4px solid #2E7D32;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }

.sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: white;
    }

    .nav-button {
        display: block;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        text-decoration: none;
        transition: background-color 0.3s;
    }

    .nav-button:hover {
        background-color: #444;
    }

    .nav-active {
        background-color: #FF4B4B;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

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

# Get Severity CSS class
def get_severity_class(severity_level):
    if severity_level == "low":
        return "severity-low"
    elif severity_level == "medium":
        return "severity-medium"
    else:
        return "severity-high"

# Sidebar
# with st.sidebar:
#     st.image("plant2.jpeg", use_container_width=True)
#     st.markdown('<div class="sidebar-title">🌿 Dashboard</div>', unsafe_allow_html=True)
#     st.markdown("---")

#     # Custom styled links
#     for page in ["Home", "About", "Disease Recognition", "Prediction History"]:
#         active_class = "nav-active" if st.session_state.selected_page == page else ""
#         st.markdown(
#             f'<a href="#" class="nav-button {active_class}" onclick="window.location.reload(true);">{page}</a>',
#             unsafe_allow_html=True
#         )

#     st.markdown("---")
#     st.markdown("""
#     **Developed by:**  
#     AI For Agriculture Team

#     **Contact:**  
#     [📧 Email](mailto:support@aiagri.com)  
#     [🌐 Website](https://aiforagri.com)
#     """)

# old
with st.sidebar:
    st.image("plant2.jpeg", use_container_width=True)
    st.markdown("# 🌿 Dashboard")
    st.markdown("---")
    
    app_mode = st.selectbox(
        "Select Page", 
        ["Home", "About", "Disease Recognition", "Prediction History"],
        key="sidebar_selectbox"
    )
    
    st.markdown("---")
    st.markdown("""
    **Developed by:**  
    AI For Agriculture benefit

    """)

# Initialize prediction history in session state if it doesn't exist
if 'pred_history' not in st.session_state:
    st.session_state.pred_history = []  # To store tuples of (prediction, image)

# Main Page
if app_mode == "Home":
    st.markdown("<h1 class='main-header'>COTTON LEAF DISEASE RECOGNITION SYSTEM</h1>", unsafe_allow_html=True)
    
    # Main image
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("plant2.jpeg", use_container_width=True)
    
    # Introduction cards
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    ## Welcome to the Cotton Leaf Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying cotton plant diseases efficiently and provide treatment recommendations. 
    Upload an image of a cotton plant leaf, and our system will analyze it to detect any signs of diseases 
    and recommend appropriate fertilizers, nutrients, and pesticides.
    Together, let's protect our cotton crops and ensure a healthier harvest!
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # How it works
    st.markdown("<h2 class='sub-header'>How It Works</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 1. Upload Image")
        st.markdown("Go to the *Disease Recognition* page and upload an image of a cotton plant leaf with suspected diseases.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 2. Analysis")
        st.markdown("Our system will process the image using advanced algorithms to identify potential diseases and their severity.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 3. Results")
        st.markdown("View the results and get recommendations for fertilizers, nutrients, and pesticides to treat the identified disease.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Why choose us
    st.markdown("<h2 class='sub-header'>Why Choose Us?</h2>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **Comprehensive:** Not just detection, but also treatment recommendations based on disease and severity.
        """)
    
    with col2:
        st.markdown("""
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
        """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Get started
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🚀 Get Started")
    st.markdown("Click on the *Disease Recognition* page in the sidebar to upload an image and experience the power of our Cotton Disease Recognition System!")
    st.markdown("</div>", unsafe_allow_html=True)

# About Project
elif app_mode == "About":
    st.markdown("<h1 class='main-header'>About Our Project</h1>", unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Project Overview")
    st.markdown("""
    Our Cotton Leaf Disease Recognition System aims to help farmers quickly identify and treat cotton plant diseases, 
    reducing crop losses and improving yields. By leveraging computer vision and machine learning, we provide accurate 
    disease identification and personalized treatment recommendations.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Dataset Information
    st.markdown("<h2 class='sub-header'>About Dataset</h2>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("plant2.jpeg", use_container_width=True)
    
    with col2:
        st.markdown("""
        This dataset consists of images of healthy and diseased cotton crop leaves categorized into 4 different classes:
        
        - **Bacterial Blight**: A bacterial disease characterized by angular, water-soaked lesions
        - **Curl Virus**: A viral disease causing leaf curling and stunted growth
        - **Fussarium Wilt**: A fungal disease causing yellowing and wilting of leaves
        - **Healthy**: Normal cotton leaves without disease symptoms
        """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Approach
    st.markdown("<h2 class='sub-header'>About Our Approach</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Disease Detection")
        st.markdown("""
        - Deep learning model trained on thousands of cotton leaf images
        - Multi-class classification with confidence scores
        - Image preprocessing for enhanced feature extraction
        - Severity estimation based on infected area analysis
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Treatment Recommendations")
        st.markdown("""
        - Customized recommendations based on disease type and severity
        - Comprehensive treatment approach including:
          - Suitable fertilizers with dosage guidance
          - Required nutrients for plant recovery
          - Appropriate pesticides with application instructions
          - Best cultural practices for disease management
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Team
    st.markdown("<h2 class='sub-header'>Our Team</h2>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    Our interdisciplinary team combines expertise in:
    - Agricultural Science
    - Machine Learning
    - Computer Vision
    - Software Development
    
    Together, we're committed to creating innovative solutions for sustainable agriculture.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown("<h1 class='main-header'>Cotton Disease Recognition</h1>", unsafe_allow_html=True)
    
    # Image upload section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Upload Cotton Leaf Image")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_image = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
    
    with col2:
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
        st.image(test_image, caption="Selected Image", use_container_width=True)
    
    predict_button = st.button("📊 Analyze Image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction results
    if predict_button:
        if test_image is not None:
            # Display a spinner while processing
            with st.spinner("Analyzing leaf image... Please wait."):
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
                
                # Calculate severity
                severity_percentage, total_leaf_pixels, infected_pixels = calculate_severity(test_image)
            
            # Display prediction results
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h2 class='sub-header'>Diagnosis Results</h2>", unsafe_allow_html=True)
            
            # Format disease name for display
            display_disease = predicted_disease.replace("_", " ").title()
            
            # Display disease prediction
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"### Detected Condition:")
                st.markdown(f"<h1 style='color:#2E7D32;'>{display_disease}</h1>", unsafe_allow_html=True)
            
            with col2:
                # Create a chart of predictions
                chart_data = pd.DataFrame({
                    'Disease': ['Bacterial Blight', 'Curl Virus', 'Fussarium Wilt', 'Healthy'],
                    'Confidence': [bacterial_blight, curl_virus, fussarium_wilt, healthy]
                })
                st.bar_chart(chart_data.set_index('Disease'))
            
            # Display detailed percentages
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Bacterial Blight", f"{bacterial_blight:.1f}%")
            col2.metric("Curl Virus", f"{curl_virus:.1f}%")
            col3.metric("Fussarium Wilt", f"{fussarium_wilt:.1f}%")
            col4.metric("Healthy", f"{healthy:.1f}%")
            
            # Severity Section
            st.markdown("<h3 class='sub-header'>Infection Severity Analysis</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                # Show the calculation steps
                st.markdown("#### Area Analysis")
                st.markdown(f"**Total Leaf Area (TLA)**: {total_leaf_pixels:,} pixels")
                st.markdown(f"**Infected Leaf Area (ILA)**: {infected_pixels:,} pixels")
                st.markdown(f"**Severity Calculation**: ({infected_pixels:,} / {total_leaf_pixels:,}) × 100 = {severity_percentage:.2f}%")
            
            with col2:
                # Display the severity result with progress bar
                st.markdown("#### Severity Indicator")
                st.progress(min(severity_percentage/100, 1.0))
                
                # Get treatment recommendations
                recommender = CottonDiseaseRecommender()
                recommendations = recommender.get_recommendations(predicted_disease, severity_percentage)
                
                # Determine severity level
                severity_level = recommendations["severity"]
                severity_class = get_severity_class(severity_level)
                
                st.markdown(f"**Severity Level**: <span class='{severity_class}'>{severity_level.upper()}</span>", unsafe_allow_html=True)
                st.info(recommendations['recommendations']['severity_guidance']['message'])
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Treatment Recommendations
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h2 class='sub-header'>Treatment Recommendations</h2>", unsafe_allow_html=True)
            
            # Create tabs for different recommendation categories
            tab1, tab2, tab3, tab4 = st.tabs(["Fertilizers", "Nutrients", "Pesticides", "Cultural Practices"])
            
            with tab1:
                st.markdown(f"**Recommended Dosage**: {recommendations['recommendations']['severity_guidance']['fertilizer_dose']}")
                for fertilizer in recommendations['recommendations']['fertilizers']:
                    st.markdown(f"<div class='recommend-item'>🌱 {fertilizer}</div>", unsafe_allow_html=True)
            
            with tab2:
                for nutrient in recommendations['recommendations']['nutrients']:
                    st.markdown(f"<div class='recommend-item'>💧 {nutrient}</div>", unsafe_allow_html=True)
            
            with tab3:
                st.markdown(f"**Recommended Dosage**: {recommendations['recommendations']['severity_guidance']['pesticide_dose']}")
                for pesticide in recommendations['recommendations']['pesticides']:
                    st.markdown(f"<div class='recommend-item'>🧪 {pesticide}</div>", unsafe_allow_html=True)
            
            with tab4:
                for practice in recommendations['recommendations']['cultural_practices']:
                    st.markdown(f"<div class='recommend-item'>🔄 {practice}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Save prediction history with recommendations
            prediction_summary = f"Disease: {predicted_disease}, Severity: {severity_percentage:.2f}%"
            save_prediction_history(prediction_summary, test_image.name, recommendations)
            
            # Additional Resources
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3 class='sub-header'>Additional Resources</h3>", unsafe_allow_html=True)
            st.markdown("""
            - Download our [Treatment Guide PDF](https://cicr.org.in/wp-content/uploads/Cotton_Crop_Protection_Strategies_2018.pdf)
            - Find agricultural experts in your area
            - Connect with other farmers facing similar issues
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.error("⚠️ Please upload an image or take a photo before analyzing.")

# Prediction History Page
elif app_mode == "Prediction History":
    st.markdown("<h1 class='main-header'>Prediction History</h1>", unsafe_allow_html=True)
    
    # Load prediction history
    prediction_history = load_prediction_history()
    
    # Show prediction history
    if not prediction_history.empty:
        for idx, row in prediction_history.iterrows():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display image
                image_path = row['image_path']
                # Check if the image file exists before displaying
                if os.path.isfile(image_path):
                    st.image(image_path, caption=f"Image {idx + 1}", use_container_width=True)
                else:
                    st.error(f"Image file '{image_path}' not found.")
            
            with col2:
                st.markdown(f"### Analysis #{idx + 1}")
                st.markdown(f"**Result**: {row['prediction']}")
                
                # Display recommendations if available
                if row['recommendations'] and row['recommendations'] != "":
                    st.markdown("**Recommendations were provided**")
                    # if st.button(f"View Details #{idx + 1}", key=f"view_details_{idx}"):
                    #     st.session_state.selected_page = "Disease Recognition"
                
                # Add timestamp (in a real app, you would save this)
                st.markdown("**Date**: April 6, 2025")
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.info("No predictions made yet. Go to the Disease Recognition page to analyze cotton leaf images.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add sample images suggestion
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Need Sample Images?")
        st.markdown("""
        If you don't have cotton leaf images available, you can download sample images to test the system:
        - [Sample Cotton Leaf Images](https://example.com/samples)
        - [Cotton Disease Dataset](https://example.com/dataset)
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>© 2025 Cotton Leaf Disease Recognition System | Developed for Agricultural Innovation</p>
</div>
""", unsafe_allow_html=True)