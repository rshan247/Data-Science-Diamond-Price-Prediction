import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def importdata():
    diamond_data = pd.read_csv("../diamonds.csv", index_col=0)
    print("Dataset: ",diamond_data.head())
    print("\nDescriptive Stats:")
    print(diamond_data.describe())
    return diamond_data


def handle_null_values(data):
    print("Checking for null values")
    if data.isna().sum().sum() == 0:
        print("No null values detected.")
    else:
        pass


def visualize_distributions(data):
    numerical_features = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    plt.figure(figsize=(12,8))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(3,3,i)
        # diamond_data[feature].hist(color='skyblue', edgecolor = 'black')
        sns.histplot(data[feature],kde=True, color='lightgreen', edgecolor = 'black')
        plt.title(f"Distribution of {feature}")
    plt.tight_layout()
    plt.show()


def encode_categorical_features(data):
    cut_mapping = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
    data['cut'] = data['cut'].map(cut_mapping)

    encoder = LabelEncoder()

    data['color'] = encoder.fit_transform(data['color'])
    data['clarity'] = encoder.fit_transform(data['clarity'])
    return data


def standardize_numerical_features(data, standardized_features):
    scaler = StandardScaler()

    data[standardized_features] = scaler.fit_transform(data[standardized_features])

    print(f"\nStandardized data set:\n{data.head()}")
    return data


def correlation_matrix(data):
    data = data.reset_index(drop=True)
    correlation_matrix = data.corr()
    print("\nCorrelation matrix: ", correlation_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title("Correlation matrix")
    plt.show()


def mutual_info_regressor(data, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    mutual_info = mutual_info_regression(x_train, y_train)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = x_train.columns
    mutual_info.sort_values(ascending=False, inplace=True)
    print("\nMutual")
    visualize_mutual_info(mutual_info)


def visualize_mutual_info(mutual_info):
    threshold = 1
    colors = ['green' if x > threshold else 'red' for x in mutual_info]
    mutual_info.plot(kind='bar', color=colors)
    plt.title("Feature selection using Mutual Info Regression")
    plt.xlabel("Features")
    plt.xticks(rotation=30)
    plt.show()


def KBest_selector(data, x, y):
    selector = SelectKBest(score_func=mutual_info_regression, k=6)
    selector.fit(x,y)

    selected_indices = selector.get_support(indices=True)

    selected_features = x.columns[selected_indices]
    print("Selected Features:", selected_features)
    return selected_features


def split_dataset(data, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .2, random_state=42)

    return x_train, x_test, y_train, y_test


def train_using_linear_regression(x_train, x_test, y_train):
    linear_model = LinearRegression()

    linear_model.fit(x_train, y_train)
    return linear_model


def predict(x_test, trained_model):
    y_pred = trained_model.predict(x_test)
    print("\nY Predicted values: ", y_pred)
    return y_pred


def cal_accuracy(y_test, y_prediction):
    mse = mean_squared_error(y_test, y_prediction)
    print("Mean Squared Error: ", mse)

    r2 = r2_score(y_test, y_prediction)
    print("R2 Score: ", r2)


def y_prediction_visualization(x_test, y_test, y_prediction):
    sns.lineplot(x=x_test['carat'], y=y_test, color='blue', label="Actual values")
    sns.lineplot(x=x_test['carat'], y=y_prediction, color='green', label="Predicted values")
    plt.title("Actual vs Predicted Values")
    plt.show()


def main():
    diamond_data = importdata()

    # Handling null values
    handle_null_values(diamond_data)

    # Visualizing dataset distributions
    visualize_distributions(diamond_data)

    # Encoding categorical values
    diamond_data = encode_categorical_features(diamond_data)

    # Standardizing numerical values
    standardized_features = [col for col in diamond_data.select_dtypes(include=[np.number]).columns if col != 'price']
    diamond_data = standardize_numerical_features(diamond_data, standardized_features)

    # visualizig correlation matrix
    correlation_matrix(diamond_data)

    # Feature Selection
    x = diamond_data.drop("price", axis=1)
    y = diamond_data['price']
    mutual_info_regressor(diamond_data, x, y)
    selected_features = KBest_selector(diamond_data, x, y)

    # Model Training
    x = diamond_data[selected_features]
    x_train, x_test, y_train, y_test = split_dataset(diamond_data, x, y)

    linear_prediction = train_using_linear_regression(x_train, x_test, y_train)

    # Model Prediction
    y_prediction = predict(x_test, linear_prediction)
    cal_accuracy(y_test, y_prediction)
    y_prediction_visualization(x_test, y_test, y_prediction)


#Calling main function
if __name__ == "__main__":
    main()


