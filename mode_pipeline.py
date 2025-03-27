import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Transformers for Ordinal and One-Hot Encoding
class OrdinalEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols, mapping):
        self.cols = cols
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        encoder = OrdinalEncoder(categories=[self.mapping[col] for col in self.cols])
        X[self.cols] = encoder.fit_transform(X[self.cols])
        return X

class OneHotEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.get_dummies(X, columns=self.cols, drop_first=True)
        return X

def main_pipeline(filepath='salaries.csv'):
    """Executes the data processing and model training pipeline using sklearn Pipeline."""

    df = pd.read_csv(filepath)
    df_model = df.drop(columns=['salary', 'salary_currency'])

    ordinal_cols = ['experience_level', 'company_size', 'remote_ratio']
    ordinal_mapping = {
        'experience_level': ['EN', 'MI', 'SE', 'EX'],
        'company_size': ['S', 'M', 'L'],
        'remote_ratio': [0, 50, 100]
    }

    onehot_cols = ['employment_type', 'job_title', 'employee_residence', 'company_location']

    pipeline = Pipeline([
        ('ordinal', OrdinalEncoderWrapper(ordinal_cols, ordinal_mapping)),
        ('onehot', OneHotEncoderWrapper(onehot_cols)),
        ('model', LinearRegression())
    ])

    X = df_model.drop(columns=['salary_in_usd'])
    y = df_model['salary_in_usd']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    pickle.dump(pipeline, open('model.pkl', 'wb'))

if __name__ == "__main__":
    main_pipeline()