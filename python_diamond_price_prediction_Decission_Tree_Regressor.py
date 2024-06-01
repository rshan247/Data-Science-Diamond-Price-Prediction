import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score



def importdata():
    diamond_data = pd.read_csv("diamonds.csv", index_col=0)
    print("Dataset:\n",diamond_data.head())
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
    print("\nCorrelation matrix:\n", correlation_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title("Correlation matrix")
    plt.show()


def feature_importance(data, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(x_train, y_train)

    importances = dt_model.feature_importances_

    feature_importance = pd.DataFrame({"Importance": importances}, index=x.columns)
    feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
    print("\nFeature importance using Random Forest:\n", feature_importance)
    visualize_feature_importance(feature_importance)

    return feature_importance.head(6).index


def visualize_feature_importance(feature_importance):
    feature_importance.plot(kind="bar", color = "green")
    plt.title("Feature selection using Decision Tree Feature Importance")
    plt.xticks(rotation=45)
    plt.show()


def split_dataset(data, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .2, random_state=42)

    return x_train, x_test, y_train, y_test


def train_using_decision_tree(x_train, x_test, y_train):
    decision_tree_model = DecisionTreeRegressor( random_state=42)

    decision_tree_model.fit(x_train, y_train)
    return decision_tree_model


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
    standardized_features = [col for col in diamond_data.columns if col != 'price' ]
    # standardized_features = [col for col in diamond_data.select_dtypes(include=[np.number]).columns if col != 'price']
    diamond_data = standardize_numerical_features(diamond_data, standardized_features)

    # visualizig correlation matrix
    correlation_matrix(diamond_data)

    # Feature Selection
    x = diamond_data.drop("price", axis=1)
    y = diamond_data['price']

    selected_features = feature_importance(diamond_data, x, y)

    # Model Training
    x = diamond_data[selected_features]
    x_train, x_test, y_train, y_test = split_dataset(diamond_data, x, y)

    decision_tree_prediction = train_using_decision_tree(x_train, x_test, y_train)

    # Model Prediction
    y_prediction = predict(x_test, decision_tree_prediction)
    cal_accuracy(y_test, y_prediction)
    y_prediction_visualization(x_test, y_test, y_prediction)


#Calling main function
if __name__ == "__main__":
    main()


