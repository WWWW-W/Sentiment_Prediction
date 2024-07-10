Sentiment Analysis Model Training and Evaluation

Project Overview

This project involves training and evaluating sentiment analysis models using different neural network architectures, such as LSTM and CNN. The models are trained on a preprocessed and augmented dataset of movie reviews and are evaluated based on their accuracy and performance metrics. The project includes hyperparameter tuning, model performance visualization, and saving the best model in both .h5 and TensorFlow Lite formats.


## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Notebook Sections](#notebook-sections)
- [Model Training with TensorFlow](#model-training-with-tensorflow)
- [Model Training Pipeline](#model-training-pipeline)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Saving the Model](#saving-the-model)


## Requirements

Python version: 3.11.7 | packaged by Anaconda, Inc. | (main, Dec 15 2023, 18:05:47) [MSC v.1916 64 bit (AMD64)]
Jupyter Notebook
TensorFlow version: 2.16.2
Keras Tuner version: 1.4.7
NLTK version: 3.8.1
Pandas version: 2.2.2
NumPy version: 1.26.4


## Usage

1. **Open the notebook:**

    Open `preprocess2.ipynb, project.ipynb, Tensorboard.ipynb` in Jupyter Notebook.

2. ## Notebook Sections

2.1preprocess2.ipynb file:
 **Data Preprocessing:**
    - Load and clean the dataset.
    - Perform data augmentation.
    - Tokenize and pad the text data.


2.2 project.ipynb file:
**Model Training with TensorFlow:**
    - Experiment with different model architectures such as feedforward neural networks, CNNs, and LSTMs.
    - Explore various hyperparameter configurations and optimization techniques.
    - Track and document experiments using TensorBoard.
on Tensorboard.ipynb file
    - Evaluate the trained model using appropriate metrics and validation techniques.

**Model Training Pipeline:**
    - Automate data preprocessing, including data cleaning, normalization, and feature engineering.
    - Integrate data augmentation techniques to enhance the model's ability to generalize.
    - Streamline the model training process using TensorFlow's high-level APIs like Keras.

**Hyperparameter Tuning:**
    - Use Keras Tuner to find the best hyperparameters.
    - Train the model with the best hyperparameters.

**Saving the Model:**
    - Save the trained model.
    - Convert the model to TensorFlow Lite format.

3. ## Model Training with TensorFlow

The model training section includes:
- Data cleaning and preprocessing steps using NLTK.
- Experimentation with model architectures (CNNs, LSTMs).
- Hyperparameter tuning to find the best configurations.
- Tracking experiments with TensorBoard.
- Evaluating model performance.
## Model Training Pipeline

The notebook demonstrates how to:
- Automate the process of data preprocessing, including cleaning, normalization.
- Implement data augmentation techniques to improve model generalization.
- Integrate the entire model training process into a pipeline for reproducibility and ease of experimentation.
- Utilize TensorFlow's high-level APIs such as Keras and TensorFlow Estimator API to streamline the development.

4. ## Hyperparameter Tuning

The notebook uses Keras Tuner to perform hyperparameter tuning. It explores various configurations to find the best set of hyperparameters for the model.

5. ## Saving the Model

After training, the model is saved in both the standard .h5 format and TensorFlow Lite format for deployment on web app.

