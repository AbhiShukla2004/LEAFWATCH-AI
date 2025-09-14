Potato Disease Classification
Project Overview
This project develops a deep learning model to classify potato diseases from images. It utilizes a Convolutional Neural Network (CNN) to distinguish between three categories: healthy potatoes and those affected by Early Blight or Late Blight.

Dataset
The model was trained and validated on a dataset of 2,152 images. The data was split into training and validation sets with an 80/20 ratio to ensure the model's performance was evaluated on unseen data, validating its generalization capabilities.

Model and Methodology
The project's core is a CNN built using the TensorFlow and Keras frameworks. The model architecture is designed to efficiently extract and learn features from the images.

Key steps in the methodology include:

Data Preprocessing: Images were resized and rescaled to a uniform size before being fed into the model.

Data Augmentation: Techniques like random rotations and flips were applied to the training data to increase its diversity and prevent the model from overfitting.

Key Results
The trained model demonstrated strong performance on the validation set:

Accuracy: 96.48%

Getting Started
To run this project, you will need to have Python installed. The following libraries are required and can be installed via pip:

Bash

pip install tensorflow
pip install keras
pip install numpy
pip install matplotlib
The model training and evaluation process is documented in the training.ipynb Jupyter notebook.
