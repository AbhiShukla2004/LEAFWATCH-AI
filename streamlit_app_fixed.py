import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Set page config
st.set_page_config(
    page_title="LeafWatch AI - Potato Disease Detection",
    page_icon="🍃",
    layout="centered"
)

# Title and description
st.title("LeafWatch AI")
st.subheader("Potato Disease Detection")
st.write("Upload an image of a potato plant leaf to detect if it's healthy or has a disease.")

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model('models/model_1.keras')
    return model

# Preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Changed to 256x256 to match training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Note: We don't need to normalize here since it's part of the model's preprocessing layers
    return img_array

# Class names in the same order as training
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

try:
    model = load_trained_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', width=300)
        st.write("Classifying...")
        
        # Save the uploaded file temporarily and preprocess
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Preprocess and predict
        processed_image = preprocess_image("temp_image.jpg")
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Clean up the display name by removing "Potato___" prefix
        display_class = predicted_class.replace('Potato___', '').replace('_', ' ').title()
        
        # Display results
        st.success(f"Prediction: {display_class}")
        st.info(f"Confidence: {confidence:.2f}%")
        
        # Display additional information based on prediction
        if 'healthy' in predicted_class.lower():
            st.balloons()
            st.write("🌿 Your potato plant appears to be healthy! Keep up the good care!")
        else:
            st.warning(f"""
            {display_class} detected! Here are some general care tips:
            - Remove infected leaves
            - Ensure proper air circulation
            - Apply appropriate fungicide
            - Monitor water management
            """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please make sure the model file exists and all requirements are installed.")