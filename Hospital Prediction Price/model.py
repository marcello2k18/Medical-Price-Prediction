import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import pickle

class MedicalInsuranceModel:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self._preprocess_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_transformed, self.y, test_size=0.2, random_state=42)
        self.regression_models = {}
        self.classification_models = {}

    def _preprocess_data(self):
        categorical_features = ['sex', 'smoker', 'region']
        one_hot = OneHotEncoder()
        self.transformer = ColumnTransformer(
            [("one_hot", one_hot, categorical_features)], remainder="passthrough")
        self.X_transformed = self.transformer.fit_transform(self.data.drop('charges', axis=1))
        self.y = self.data['charges']

    def train_regression_models(self):
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regression": RandomForestRegressor(),
            "Gradient Boosting Regression": GradientBoostingRegressor(),
            "K-Nearest Neighbors Regression": KNeighborsRegressor(),
            "XGBoost Regression": XGBRegressor(),
            "LightGBM Regression": LGBMRegressor(),
            "CatBoost Regression": CatBoostRegressor(),
            "Support Vector Regression": SVR()
        }
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            self.regression_models[name] = model

    def train_classification_models(self):
        self.data['charges_category'] = (self.data['charges'] > self.data['charges'].median()).astype(int)
        y_categorical = self.data['charges_category']
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
            self.X_transformed, y_categorical, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN Classification": KNeighborsClassifier(),
            "SVM Classification": SVC(probability=True),
            "Naive Bayes Classification": GaussianNB(),
            "XGBoost Classification": XGBClassifier()
        }
        for name, model in models.items():
            model.fit(X_train_cat, y_train_cat)
            self.classification_models[name] = model

    def predict_regression(self, user_input):
        user_data = pd.DataFrame([user_input], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        user_data_transformed = self.transformer.transform(user_data)
        predictions = {}
        for name, model in self.regression_models.items():
            prediction = model.predict(user_data_transformed)[0]
            predictions[name] = prediction
        return predictions

    def predict_classification(self, user_input):
        user_data = pd.DataFrame([user_input], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        user_data_transformed = self.transformer.transform(user_data)
        predictions = {}
        for name, model in self.classification_models.items():
            prediction = model.predict(user_data_transformed)[0]
            pred_class = "High charges" if prediction == 1 else "Low charges"
            predictions[name] = pred_class
        return predictions

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
