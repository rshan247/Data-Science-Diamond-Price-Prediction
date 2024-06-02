import numpy as np
import joblib
from flask import Flask , request, jsonify
from flask_cors import CORS
from collections import OrderedDict


app = Flask(__name__)
CORS(app)


# Load the model
random_forest_model = joblib.load("Saved_model_files/random_forest_model.pkl")
means = np.load('Saved_model_files/means.npy')
stds = np.load('Saved_model_files/stds.npy')

def standardize_user_input(user_input, means_array, stds_array):
    return (user_input - means_array) / stds_array

def predict_price(user_input):

    user_input_standardized = standardize_user_input(user_input, means, stds)
    result = random_forest_model.predict([user_input_standardized])
    return result
    print(f"The predicted price for the inputted diamond attributes is {result}")


@app.route('/predict', methods = ['POST'])
def predict():
    try:
        user_input = dict(request.json['data'])
        key_order = ['carat', 'y', 'clarity', 'color', 'z', 'x']
        ordered_user_input = dict([(k, user_input[k]) for k in key_order])
        ordered_user_input_list = list(ordered_user_input.values())
        price = predict_price(list(ordered_user_input.values()))
        return jsonify({'price' : price[0]})
    except Exception as e:
        print("exception: ",e)
        return jsonify({'error': str(e)}), 400
user_input = np.array([.83, 5.98, 3, 1, 4.43, 3.95])


if __name__ == '__main__':
    app.run(debug=True)



"""
color:
0 - D
1 - E
2 - F
3 - G
4 - H
5 - I
6 - J

Clarity:
0 - I1
1 - IF
2 - SI1
3 - SI2
4 - VS1
5 - VS2
6 - VVS1
7 - VVS2
"""