"""
Pneumonia Detection Web App
============================
This is a Streamlit web application that allows users to upload chest X-ray images
and get instant pneumonia detection predictions using the trained AI model.

Author: FYP Abdul Awan
Model: ResNet50 Transfer Learning
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import csv
from datetime import datetime

# ====================================
# PAGE CONFIGURATION
# ====================================
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="centered"
)

# ====================================
# LOAD MODEL (Cached for performance)
# ====================================
@st.cache_resource
def load_pneumonia_model():
    """Load the trained model once and cache it"""
    try:
        model = load_model('pneumonia_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure 'pneumonia_model.h5' is in the same folder as app.py")
        return None

# ====================================
# IMAGE PREPROCESSING
# ====================================
def preprocess_image(image):
    """
    Preprocess the uploaded image for model prediction
    - Resize to 224x224 (model input size)
    - Convert to RGB if needed
    - Normalize pixel values to 0-1
    - Add batch dimension
    """
    # Convert to RGB if image has alpha channel or is grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values to 0-1
    img_array = img_array / 255.0
    
    # Add batch dimension (model expects batch of images)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ====================================
# PREDICTION LOGGING
# ====================================
def log_prediction(filename, prediction_label, confidence):
    """Log user predictions to CSV file for tracking"""
    log_dir = 'user_predictions'
    log_file = os.path.join(log_dir, 'predictions_log.csv')
    
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create CSV with header if file doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'filename', 'prediction', 'confidence'])
    
    # Append prediction to log
    try:
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                filename,
                prediction_label,
                f"{confidence:.1f}%"
            ])
    except Exception as e:
        # Silently fail if logging doesn't work (doesn't affect user experience)
        pass

# ====================================
# MAIN APP
# ====================================
def main():
    # Title and description
    st.title("🫁 Pneumonia Detection System")
    st.write("Upload a chest X-ray image to detect pneumonia using AI")
    
    # Add some space
    st.markdown("---")
    
    # Load the model
    model = load_pneumonia_model()
    
    if model is None:
        st.stop()  # Stop if model couldn't be loaded
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (JPG, PNG, JPEG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a chest X-ray image for pneumonia detection"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Load and display the image
            image = Image.open(uploaded_file)
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded X-Ray")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Analysis")
                
                # Add a processing message
                with st.spinner('Analyzing X-ray...'):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    prediction = model.predict(processed_image, verbose=0)
                    prediction_value = prediction[0][0]
                    
                    # Interpret prediction
                    # Model outputs: 0 = NORMAL, 1 = PNEUMONIA
                    if prediction_value > 0.5:
                        prediction_label = "PNEUMONIA"
                        confidence = prediction_value * 100
                        st.error(f"⚠️ **{prediction_label} DETECTED**")
                        st.metric("Confidence", f"{confidence:.1f}%")
                        st.warning("**Recommendation:** Please consult a medical professional immediately for confirmation and treatment.")
                    else:
                        prediction_label = "NORMAL"
                        confidence = (1 - prediction_value) * 100
                        st.success(f"✅ **{prediction_label}**")
                        st.metric("Confidence", f"{confidence:.1f}%")
                        st.info("**Note:** No signs of pneumonia detected. However, if you have symptoms, please consult a doctor.")
                
                # Log the prediction
                log_prediction(uploaded_file.name, prediction_label, confidence)
            
            # Additional information
            st.markdown("---")
            st.markdown("""
            ### ℹ️ About This System
            
            This AI system uses deep learning (ResNet50 architecture) to analyze chest X-ray images 
            and detect signs of pneumonia with **86.4% accuracy**.
            
            **Important:**
            - This is an AI assistance tool, not a replacement for professional medical diagnosis
            - Always consult qualified healthcare professionals for medical decisions
            - This system is for educational and research purposes (FYP Project)
            
            **How it works:**
            1. Upload a chest X-ray image
            2. AI analyzes the image using a trained neural network
            3. Prediction is shown with confidence score
            """)
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.info("Please make sure you uploaded a valid image file.")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a chest X-ray image to get started")
        
        # Show example/demo section
        with st.expander("ℹ️ How to use this system"):
            st.markdown("""
            **Step 1:** Click the "Browse files" button above
            
            **Step 2:** Select a chest X-ray image from your computer
            
            **Step 3:** Wait a few seconds for the AI to analyze
            
            **Step 4:** View the prediction result and confidence score
            
            **Note:** This system works best with clear, frontal chest X-ray images
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Pneumonia Detection System | FYP Project | Powered by AI</p>
    <p>Model Accuracy: 86.4% | Using ResNet50 Transfer Learning</p>
    </div>
    """, unsafe_allow_html=True)

# ====================================
# RUN APP
# ====================================
if __name__ == "__main__":
    main()