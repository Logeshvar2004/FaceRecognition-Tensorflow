import streamlit as st
import tensorflow as tf
import io
import numpy as np
from PIL import Image

def load_model():
    return tf.keras.models.load_model(r'D:\Coding\python\MELE\face_recognition_model.h5')

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the image
    return image


def load_and_preprocess_image(uploaded_file):
    # Read the uploaded file as an Image object
    image = Image.open(uploaded_file)

    # Resize the image to the desired dimensions
    image = image.resize((128, 128))

    # Convert the image to a NumPy array
    image = np.array(image)

    # Normalize the pixel values to [0, 1]
    image = image / 255.0

    return image


# Main function to run the app
def main():
    st.title('Face Recognition Model Test')

    # File uploader
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg'])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Preprocess the uploaded image and run inference
        image = load_and_preprocess_image(uploaded_file)
        predictions = model.predict(tf.expand_dims(image, axis=0))

        # Get the predicted class index
        predicted_class = tf.argmax(predictions, axis=1)

        # Get the confidence score
        confidence_score = tf.reduce_max(predictions)

        # Define a dictionary mapping class indices to celebrity names
        predicted_class = int(tf.argmax(predictions, axis=1)[0])

        # Define a dictionary mapping class indices to celebrity names
        class_to_celebrity = {
            0: "Angelina Jolie",
            1: "Brad Pitt",
            2: "Denzel Washington",
            3: "Hugh Jackman",
            4: "Jennifer Lawrence"
        }

        # Get the predicted celebrity name
        predicted_celebrity = class_to_celebrity[predicted_class]

        # Format the confidence score to percentage
        confidence_percentage = confidence_score * 100

        # Output the prediction result
        st.write(f'Predicted Celebrity: {predicted_celebrity}, Confidence: {confidence_percentage:.2f}%')

if __name__ == '__main__':
    main()
