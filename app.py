import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import matplotlib.pyplot as plt
import os
from PIL import Image

# Load precomputed outfit features
outfit_features = pkl.load(open('outfit_df.pkl', 'rb'))

# Ensure features are properly loaded and converted to a numpy array
if isinstance(outfit_features['features'].iloc[0], str):
    outfit_features['features'] = outfit_features['features'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
else:
    outfit_features['features'] = outfit_features['features'].apply(np.array)

# Convert the features column to a numpy array
outfit_features_array = np.vstack(outfit_features['features'].values)

# Extract image IDs for reference
outfit_image_ids = outfit_features['id'].values

# Fit the KNN model for outfits
outfit_knn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(outfit_features_array)

# Load precomputed jewelry features
jewelry_features = pkl.load(open('jewelry_df.pkl', 'rb'))

# Extract features and filenames
jewelry_features_array = np.array(jewelry_features['features'].tolist())
jewelry_filenames = jewelry_features['filenames'].tolist()

# Fit the KNN model for jewelry
jewelry_knn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(jewelry_features_array)

# Function to extract features using ResNet50
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        
        resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        features = resnet_model.predict(img_data)
        return features.flatten()
    except Exception as e:
        st.error(f"Error processing {img_path}: {e}")
        return None

# Function to recommend outfit images
def recommend_outfit_images(input_image_path, knn_model, features, image_ids, images_folder_path):
    input_image_features = extract_features(input_image_path)
    recommended_image_paths = []

    if input_image_features is not None:
        distances, indices = knn_model.kneighbors([input_image_features])
        recommended_image_ids = [image_ids[idx] for idx in indices[0]]
        for image_id in recommended_image_ids:
            image_path = os.path.join(images_folder_path, f'{image_id}.jpg')
            if os.path.exists(image_path):
                recommended_image_paths.append(image_path)
            else:
                st.error(f"File not found: {image_path}")
    else:
        st.error("Error extracting features from the input image.")
    
    return recommended_image_paths

# Function to recommend jewelry images
def recommend_jewelry_images(input_image_path, knn_model, features, filenames, images_folder_path):
    input_image_features = extract_features(input_image_path)
    recommended_image_paths = []

    if input_image_features is not None:
        distances, indices = knn_model.kneighbors([input_image_features])
        recommended_filenames = [filenames[idx] for idx in indices[0]]
        for filename in recommended_filenames:
            image_path = os.path.join(images_folder_path, filename)
            if os.path.exists(image_path):
                recommended_image_paths.append(image_path)
            else:
                st.error(f"File not found: {image_path}")
    else:
        st.error("Error extracting features from the input image.")
    
    return recommended_image_paths

# Streamlit UI
st.set_page_config(page_title="Fashion Stylist", page_icon=":dress:", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>Fashion Stylist Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #FF69B4;'>Discover Your Next Favorite Outfit and Jewelry</h3>", unsafe_allow_html=True)

# Create directory to save uploaded images
os.makedirs("uploaded_images", exist_ok=True)

# Upload outfit image
uploaded_file = st.file_uploader("Choose an outfit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image
    input_image_path = os.path.join("uploaded_images", uploaded_file.name)
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    st.image(input_image_path, caption='Uploaded Image.', use_column_width=False, width=300)
    st.write("")
    
    # Recommend similar outfits
    st.write("## Similar Outfits")
    images_folder_path = r'F:\fashion.csv\imagess'  # Update this path
    recommended_outfit_paths = recommend_outfit_images(input_image_path, outfit_knn, outfit_features_array, outfit_image_ids, images_folder_path)
    
    # Display recommended outfits
    cols = st.columns(5)
    for i, image_path in enumerate(recommended_outfit_paths):
        with cols[i % 5]:
            st.image(image_path, use_column_width=False, width=100)
    
    # Option to recommend jewelry
    if st.button("Recommend Jewelry"):
        st.write("## Recommended Jewelry")
        jewelry_images_folder_path = r'F:\fashion.csv\jwellery_data\explo'  # Update this path
        recommended_jewelry_paths = recommend_jewelry_images(input_image_path, jewelry_knn, jewelry_features_array, jewelry_filenames, jewelry_images_folder_path)
        
        # Display recommended jewelry
        cols = st.columns(5)
        for i, image_path in enumerate(recommended_jewelry_paths):
            with cols[i % 5]:
                st.image(image_path, use_column_width=False, width=150)