# **Digit Recognition**

This project demonstrates a machine learning model built using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The dataset contains 28x28 pixel images of handwritten digits, and the model is trained to predict the correct digit label based on these images. The application performs data loading, normalization, model training, evaluation, and prediction.

![TensorFlow Badge](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python Badge](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## Features

- **MNIST Dataset**: Uses the MNIST dataset containing images of handwritten digits (0-9) for training and testing.
- **Model Training**: Builds and trains a neural network model to classify the digits.
- **Data Normalization**: Normalizes image data to improve model training.
- **Evaluation**: Evaluates the model on both training and testing datasets to check accuracy.
- **Random Prediction**: Makes random predictions and compares them with actual labels to demonstrate the model's performance.

## Technology Stack

- **Framework**: TensorFlow
- **Dataset**: MNIST (from TensorFlow Datasets)
- **Modeling**: Keras (part of TensorFlow)
- **Visualization**: Matplotlib for visualizing results

## How it Works

1. **Data Loading**: The MNIST dataset is loaded using TensorFlow Datasets (TFDS), split into training and testing datasets.
2. **Normalization**: The images are normalized by casting to 32-bit floats and scaling pixel values to a range between 0 and 1.
3. **Model Building**: A simple feedforward neural network model is built using Keras, consisting of a Flatten layer, a Dense layer with 128 neurons, and a Dense output layer for 10 classes (digits 0-9).
4. **Model Training**: The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function for 8 epochs.
5. **Evaluation**: The model is evaluated on both the training and testing datasets.
6. **Prediction**: A random prediction is made on the test dataset, and the model’s prediction is compared with the actual label.

## Results

- **Training vs. Testing Accuracy**: You might observe that the accuracy on the training dataset is higher than the testing dataset. This discrepancy is often due to **overfitting**, where the model performs well on training data but does not generalize as well to unseen test data.

## Code Explanation

### Loading the MNIST Dataset
```python
(dataset_train, dataset_test), dataset_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
