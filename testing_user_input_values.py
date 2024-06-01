import numpy as np
import joblib


def standardize_user_input(user_input, means_array, stds_array):
    return (user_input - means_array) / stds_array

def calculate_user_input(user_input):

    # Load the model
    random_forest_model = joblib.load("Saved_model_files/random_forest_model.pkl")
    means = np.load('Saved_model_files/means.npy')
    stds = np.load('Saved_model_files/stds.npy')

    user_input_standardized = standardize_user_input(user_input, means, stds)
    result = random_forest_model.predict([user_input_standardized])
    print(f"The predicted price for the inputted diamond attributes is {result}")



user_input = np.array([.83, 5.98, 3, 1, 4.43, 3.95])
# Calculate user input
calculate_user_input(user_input)