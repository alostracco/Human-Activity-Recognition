# Human Activity Recognition (HAR)

This project compares the effectiveness of different machine learning models, specifically Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks, for performing Human Activity Recognition (HAR) tasks. The goal is to determine the most efficient model for classifying human activities based on time-series data from wearable accelerometer sensors.

## Installation
To get started with this project, follow the steps below to clone the repository and install the necessary dependencies:
 - git clone https://github.com/alostracco/Human-Activity-Recognition.git
 - cd Human-Activity-Recognition
 - pip install -r requirements.txt

## Features
Activity Classification: Classifies six human activities:
 - Walking
 - Sitting
 - Standing
 - Jogging
 - Ascending stairs
 - Descending stairs

Time-Series Dataset: The dataset consists of 3D positional values from wearable accelerometer sensors (Dataset retreived from: https://www.kaggle.com/datasets/die9origephit/human-activity-recognition/data).

Machine Learning Models:
 - RNN: A simple Recurrent Neural Network model for time-series classification.
 - LSTM: A more advanced Long Short-Term Memory network designed to capture long-term dependencies in sequential data.