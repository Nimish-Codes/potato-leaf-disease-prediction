import streamlit as st
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle

# Load the pickled model
with open('potato_leaf_disease.pkl', 'rb') as f:
    model = pickle.load(f)


def main():

    st.title("Potatoo leaf disease prediction")
    st.warning("Upload only unhealthy(Early blight or Late blight) single potato leaf image")
    
    # used model:
    image_path = st.file_uploader("choose or upload an image of unhealthy potato leaf", type = ["jpg", "jpeg", "png", "webp"])

    if image_path is not None:
  
      img_height, img_width = 150, 150
  
      # Prediction function
      def predict_leaf_disease(image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        class_labels = ['Early_Blight', 'Healthy', 'Late_Blight']
        predicted_class = class_labels[np.argmax(prediction)]
        return predicted_class
    
      predicted_class = predict_leaf_disease(image_path)
      st.success(f'Predicted disease: {predicted_class}')

if __name__ == '__main__':
  main()
