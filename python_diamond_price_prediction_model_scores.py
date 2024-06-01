import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

"""--------------------------Reading the data--------------------------"""

diamond_data = pd.read_csv("diamonds.csv")
print(diamond_data.head())

numerical_features = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']



"""-----------------------Encoding Categorical Features--------------------"""

cut_mapping = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
diamond_data['cut'] = diamond_data['cut'].map(cut_mapping)

encoder = LabelEncoder()

diamond_data['color'] = encoder.fit_transform(diamond_data['color'])
diamond_data['clarity'] = encoder.fit_transform(diamond_data['clarity'])


"""-------------------Standardizing the Numerical features----------------"""

scaler = StandardScaler()

standardized_features = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']

diamond_data[standardized_features] = scaler.fit_transform(diamond_data[standardized_features])

print(f"\nStandardized data set:\n{diamond_data.head()}")



"""----------------------------Feature Selection--------------------------"""
x = diamond_data[standardized_features]
y = diamond_data['price']


selector = SelectKBest(score_func=mutual_info_regression, k=6)
X_selected = selector.fit_transform(x, y)

selected_indices = selector.get_support(indices=True)


# Print the selected feature names
selected_features = x.columns[selected_indices]
print("Selected Features:", selected_features)


"""----------------------------Model scoring-----------------------------"""
# selected_features = ['carat', 'x','y', 'z', 'table', 'depth', 'cut_encoded']
x = diamond_data[selected_features]
y = diamond_data['price']


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# models = {
#     "Linear Regression": LinearRegression(),
#     "Decision Tree": DecisionTreeRegressor(),
#     "Random Forest": RandomForestRegressor(),
#     "Gradient Boosting": GradientBoostingRegressor(),
#     "KNN Regressor": KNeighborsRegressor()
# }

models = {
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

kf = KFold(n_splits=10, shuffle=True, random_state=10)

for name, model in models.items():
    r2_scores = cross_val_score(model, x, y, cv=5, scoring='r2')
    print(f"R2 Score: {name} - Mean R-squared: {r2_scores.mean()} - Std: {r2_scores.std()}")

    cv_scores = -cross_val_score(model, x, y, cv=kf, scoring='neg_mean_squared_error')
    print(f"Neg mean squared error: {name} : {np.sqrt(cv_scores.mean())}")

    explained_variance_scores = cross_val_score(model, x, y, cv=kf, scoring='explained_variance')
    print(f"Explained Variance - Mean: {name} :{explained_variance_scores.mean()} - Std: {explained_variance_scores.std()}")

    print("\n")