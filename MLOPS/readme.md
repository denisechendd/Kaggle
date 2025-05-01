***This folder contains several Python scripts that demonstrate various aspects of Machine Learning Operations (MLOPS) such as data preprocessing, model training, and deployment.*** 

1. DashApp_example.py
Purpose: This script sets up and runs a web application using the Dash framework for predicting real estate prices based on user inputs.
Key Features:
Loads a real estate dataset and trains a linear regression model.
Creates an interactive web interface using Dash, allowing users to input features like distance to the nearest MRT station, number of convenience stores, latitude, and longitude to predict house prices.
Includes callback functions for interactivity and real-time predictions.
Source: DashApp_example.py
2. PreProcess_DAG.py
Purpose: Implements an MLOps pipeline using Apache Airflow for data preprocessing and model training.
Key Features:
Loads and preprocesses a dataset containing app usage behavior.
Performs feature engineering, such as scaling numerical features and encoding categorical variables.
Defines an Airflow DAG for automating the preprocessing steps on a daily schedule.
Source: PreProcess_DAG.py
3. resnet.py
Purpose: Implements ResNet (Residual Network) models for deep learning.
Key Features:
Contains definitions for various ResNet architectures such as ResNet-18, ResNet-50, and ResNet-101.
Provides the core building blocks (BasicBlock and Bottleneck) for constructing residual networks.
Includes pretrained model URLs for ImageNet weights.
Source: resnet.py
4. train.py
Purpose: Trains a BPR (Bayesian Personalized Ranking) model for recommendation systems.
Key Features:
Implements dataset preparation and loss function for a triplet-based ranking model.
Supports evaluation metrics like precision and recall at various thresholds.
Allows for periodic saving of the trained model and logging of training metrics.
Source: train.py
5. transform.py
Purpose: Provides utility functions for image transformation, resizing, and preprocessing for general object detection models.
Key Features:
Normalizes and resizes images to match model requirements.
Supports additional transformations like resizing bounding boxes and keypoints for object detection tasks.
Includes batching utilities for efficient processing of multiple images.
Source: transform.py

