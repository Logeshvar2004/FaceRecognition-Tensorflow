
**Face Recognition Model Training**

This project demonstrates how to train a simple face recognition model using TensorFlow. The model is trained on a dataset containing images of different celebrities.

Additionally, In this project, I used Streamlit to create a simple web interface for testing the face recognition model. Users can upload an image, and the model will predict the celebrity in the image along with the confidence score.

**Streamlit Interface:** 

![image](https://github.com/Logeshvar2004/FaceRecognition-Tensorflow/assets/102981016/caab9ad7-bda1-4254-b0aa-11dfd5e1fc32)

**Dataset Link** : https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset/data

Here I am using 5 classes: Angelina Jolie, Brad Pitt, Denzel Washington, Hugh Jackman, Jennifer Lawrence

**How It Works**

1. **Data Preparation**: The dataset is organized into subfolders, each containing images of a different celebrity so i am using Tensorflow's ImageDataGenerator module to read the images.

![image](https://github.com/Logeshvar2004/FaceRecognition-Tensorflow/assets/102981016/9d0b9348-5306-4aa3-ac0f-dcf07490fa4d)


3. **Model Architecture**: We use a Convolutional Neural Network (CNN) architecture, which is commonly used for image recognition tasks. The CNN consists of convolutional layers, pooling layers, and fully connected layers.

![image](https://github.com/Logeshvar2004/FaceRecognition-Tensorflow/assets/102981016/7d503b1f-b116-40e2-89eb-d92a783326b3)


4. **Training**: The model is trained using the images from the dataset. We use an optimizer called Adam and a loss function called categorical crossentropy to optimize the model parameters.

5. **Evaluation**: The model got 95% accuracy while training

![image](https://github.com/Logeshvar2004/FaceRecognition-Tensorflow/assets/102981016/8b7b1eb7-d224-4a00-8f41-90efa20ee8b4)



I Tested the trained face recognition model by providing an image file path. 
Image provided :

![image](https://github.com/Logeshvar2004/FaceRecognition-Tensorflow/assets/102981016/8873b518-a00d-40e1-a631-f652cd9fe0bb)


Output:

![image](https://github.com/Logeshvar2004/FaceRecognition-Tensorflow/assets/102981016/67da51f3-4648-493b-9810-f3f1bc786dca)


**Dependencies**

- TensorFlow
- Matplotlib
