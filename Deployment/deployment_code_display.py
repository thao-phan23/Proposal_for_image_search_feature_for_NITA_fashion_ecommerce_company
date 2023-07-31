
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import json
import glob
import os
import shutil
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from keras.models import load_model
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
from keras.preprocessing import image
from io import BytesIO


import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
from keras.models import load_model
from keras.preprocessing import image
from io import BytesIO

# Custom CSS styling
custom_css = """
<style>
body {
    /* Change background color to light gray */
    background-color: #f0f0f0;
    /* Change font size to 18px */
    font-size: 18px;
    /* Change font color to dark blue */
    color: #00008B;
}
/* Style the file uploader button */
div.stFileUploader > div > label {
    background-color: #f0f0f0;
    color: #333333;
    padding: 8px 12px;
    border-radius: 5px;
    font-weight: bold;
    cursor: pointer;
}
div.stFileUploader > div > label:hover {
    background-color: #e0e0e0;
}
</style>
"""

# Apply the custom CSS styling
st.markdown(custom_css, unsafe_allow_html=True)


def contrastive_loss(y_true, y_pred):
    """Calculates the contrastive loss.

    Arguments:
        y_true: List of labels, each label is of type float32.
        y_pred: List of predictions of same length as of y_true,
                each label is of type float32.

    Returns:
        A tensor containing contrastive loss as floating point value.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    margin  = 1
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean(
        (1 - y_true) * square_pred + (y_true) * margin_square)



def preprocess_image(image, target_size):
    # Check if the input is an image file path or a NumPy array
    if isinstance(image, str):  # If it's a file path
        # Read the image from the file path
        cv_image = cv2.imread(image)
        # Convert BGR to RGB (OpenCV uses BGR by default)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    else:  # If it's already a NumPy array (loaded image)
        rgb_image = image

    # Resize the image to the target size
    resized_image = cv2.resize(rgb_image, target_size)
    # Convert the image to a numpy array and normalize pixel values
    image_array = np.array(resized_image) / 255.0
    # Expand the dimensions to create a batch of size 1
    image_array = np.expand_dims(image_array, axis=0)
    return image_array



# Define function to calculate Euclidean distances:
def calculate_euclidean_distance(image1, image2_paths, target_size, is_array=False):
    if is_array:
        # If the image is already a processed array, no need to preprocess it
        image1_batch = image1
    else:
        # Preprocess the image if it's a file path
        image1_batch = preprocess_image(image1, target_size)

    distances = []
    for image2_path in image2_paths: 
        if not image2_path.endswith('.DS_Store'):  # Skip .DS_Store files
            image2_batch = preprocess_image(image2_path, target_size)
            # Predict the similarity score (Euclidean distance) between the images
            # The model returns a tuple (distance, _), so we need to extract the distance value
            distance = siamese_model_loaded.predict([image1_batch, image2_batch])[0][0]
            distances.append(distance)

    return distances




# Loaded the image classification model:
label_model_loaded = load_model('/Users/thaophan/Documents/Flatiron/Phase5_Project/cnn_model_50epoch.h5')
siamese_model_loaded = load_model('/Users/thaophan/Documents/Flatiron/Phase5_Project/siamese_model2_20epoch.h5', custom_objects={'contrastive_loss': contrastive_loss})


# Streamlit app code
st.title('Hi! Welcome to NICEON website!')
st.header('Tell us what you are looking for')


# Create a box to import an image
uploaded_file = st.file_uploader('Import your image here:', type=['jpg', 'jpeg', 'png'])

# Check if an image is uploaded
if uploaded_file is not None:
    # Convert the uploaded image to an in-memory file-like object
    image_bytes = uploaded_file.read()
    loaded_image = BytesIO(image_bytes)

    # Convert bytes to a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the NumPy array to a BGR image
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert BGR to RGB (OpenCV uses BGR by default)
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Resize the uploaded image to a smaller size
    desired_width = 256
    # Calculate the height to maintain the aspect ratio
    height, width, _ = rgb_image.shape
    aspect_ratio = width / height
    desired_height = int(desired_width / aspect_ratio)
    smaller_image = cv2.resize(rgb_image, (desired_width, desired_height))

    # Convert the smaller image back to a PIL Image object
    pil_image = Image.fromarray(smaller_image)

    # Display the uploaded image
    st.image(pil_image, caption='Uploaded Image')

    # Preprocess the uploaded image
    target_size = (256, 256)
    processed_loaded_image = preprocess_image(rgb_image, target_size)
    
    loaded_image_class_pred_probabilities = label_model_loaded.predict(processed_loaded_image)

#3. Predict the image class:

    # Get the predicted class (index of the class with the highest probability)
    loaded_image_predicted_class_index = np.argmax(loaded_image_class_pred_probabilities)

    class_labels = ['t-shirts', 'shirts', 'casual shoes', 'watches', 'sports shoes', 'kurtas', 'handbags', 'heels', 'sunglasses', 'wallets']
    loaded_image_predicted_class_label = class_labels[int(loaded_image_predicted_class_index)]

    st.write(f"Look like you are looking for {loaded_image_predicted_class_label}")

    # 4. Show top 5 similar images:
    source_image_dir = '/Users/thaophan/Documents/Flatiron/Phase5_Project/Deployment/Source_image'
    all_image_paths = []

    for class_name in os.listdir(source_image_dir):
        if class_name == '.DS_Store':
            continue
        class_dir = os.path.join(source_image_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                all_image_paths.append(image_path)

     # Calculate the Euclidean distance between the loaded image and images in the source folder
    euclidean_distances = calculate_euclidean_distance(processed_loaded_image, all_image_paths, target_size, is_array=True)

# Convert distance_list to a numpy array
    distances = np.array(euclidean_distances)

# Flatten the array to remove the extra dimension
    distances = distances.flatten()

# Get the indices of the top 5 smallest distances
    top_5_indices = np.argsort(distances)[:5]

# Extract the top 5 smallest distances using the indices
    top_5_smallest_distances = distances[top_5_indices]

# Get the paths of the top 5 most similar images
    top_5_similar_images = [all_image_paths[i] for i in top_5_indices]


    st.write(f"Here are the top 5 similar products")
    # Create a row layout with 5 columns to display the images in a single line
    col1, col2, col3, col4, col5 = st.columns(5)

# Inside the for loop to display similar images
    for idx, image_path in enumerate(top_5_similar_images):
        similar_image = Image.open(image_path)
        # Resize the image to a smaller size 
        resized_image = similar_image.resize((128, 128))

        # Display the image in the corresponding column
        if idx % 5 == 0:
            col1.image(resized_image, use_column_width=True)
        elif idx % 5 == 1:
            col2.image(resized_image, use_column_width=True)
        elif idx % 5 == 2:
            col3.image(resized_image, use_column_width=True)
        elif idx % 5 == 3:
            col4.image(resized_image, use_column_width=True)
        else:
            col5.image(resized_image, use_column_width=True)
    
    
