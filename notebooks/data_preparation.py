import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def prepare_data_for_models(
    data_path='../data/segmented_data.npy', 
    labels_path='../data/segmented_labels.csv', 
    test_size=0.2,
    window_size=100
):
    # Load segmented data from previous preprocessing
    X = np.load(data_path)
    labels_df = pd.read_csv(labels_path)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Prepare features and labels
    X = X.reshape(-1, window_size, 3)  # Ensure 3D shape: [samples, timesteps, features]
    y = labels_df['activity'].values

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42, 
        stratify=y
    )

    return X_train, X_test, y_train, y_test