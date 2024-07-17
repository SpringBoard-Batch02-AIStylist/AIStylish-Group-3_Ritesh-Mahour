**Outfit & Jewelry Recommendation System**

**Overview :**

This project aims to develop a outfit & jewelry recommendation system that matches jewelry pieces with women's outfits. The system uses a combination of ResNet50 for image feature extraction and a K-Nearest Neighbors (KNN) model for recommendations. The primary goal is to enhance the shopping experience by suggesting complementary jewelry for a given outfit.

**Features**

**Image Feature Extraction:** Utilizes ResNet50, a pre-trained convolutional neural network, to extract deep features from images of outfits and jewelry.

**Color Feature Extraction:** Computes the average color of each image to incorporate color information into the recommendation process.

**K-Nearest Neighbors (KNN) Model:** Employs KNN to find the most similar jewelry items based on combined features (deep features + color features).

**Batch Processing:** Handles large datasets by processing images and features in smaller batches to manage memory usage efficiently.

**Datasets:**

**Fashion Dataset:** Contains images and metadata of various  outfits. The images are named using unique IDs, and their paths are constructed dynamically.

**Jewelry Dataset:** Contains images and extracted features of different types of jewelry. Each image is named according to its unique ID, corresponding to different jewelry types like Black_basemetal, Blue_pearl, Gold_silver, etc.

**Implementation Details:**

Step 1: Load Required Libraries and Prepare the Datasets, Load the necessary libraries for image processing, machine learning, and deep learning.Filter the dataset to include only women's outfits.Construct image paths for the outfits based on unique IDs.

Step 2: Feature Extraction
ResNet50 Features: Use ResNet50 to extract deep features from images. These features capture the high-level representations of the images.
Color Features: Extract the average color of each image to incorporate color information into the features.

Step 3: Combine Features
Combine the deep features from ResNet50 with the color features to form a comprehensive feature vector for each image.

Step 4: Train KNN Model
Train a KNN model using the combined features of the women's outfits. This model is used to find the most similar items for recommendation.

Step 5: Recommend Jewelry
For a given input outfit, extract its features (both deep and color).
Use the trained KNN model to find the nearest neighbors from the jewelry dataset.
Recommend the top jewelry items that match the input outfit based on the combined features.

**Dependencies:**

TensorFlow/Keras

OpenCV

NumPy

Pandas

Scikit-learn

Matplotlib
 

**Acknowledgements:**

The ResNet50 model is pre-trained on the ImageNet dataset and provided by Keras.

Special thanks to the contributors of the datasets used in this project.
