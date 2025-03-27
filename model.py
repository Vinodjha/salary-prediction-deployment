import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import statsmodels.api as sm
import pickle

# Data loading
df = pd.read_csv('salaries.csv')
print(df)

#Data preparation
# Drop unnecessary columns
df_model = df.drop(columns=['salary', 'salary_currency'])  # Exclude redundant salary columns

# Define ordinal encoding for ordered features
ordinal_cols = ['experience_level', 'company_size', 'remote_ratio']
ordinal_mapping = {
    'experience_level': ['EN', 'MI', 'SE', 'EX'],
    'company_size': ['S', 'M', 'L'],
    'remote_ratio': [0, 50, 100]
}

# Apply ordinal encoding
ordinal_encoder = OrdinalEncoder(categories=[ordinal_mapping[col] for col in ordinal_cols])
df_model[ordinal_cols] = ordinal_encoder.fit_transform(df_model[ordinal_cols])

# Apply one-hot encoding to nominal features
onehot_cols = ['employment_type', 'job_title', 'employee_residence', 'company_location']
df_model = pd.get_dummies(df_model, columns=onehot_cols, drop_first=True)  # Avoid multicollinearity

# Features (X) and Target (y)
X = df_model.drop(columns=['salary_in_usd'])
y = df_model['salary_in_usd']

# Final dataset shape
print(f"Final dataset shape: {X.shape}")
X = X.astype('float32')
y= y.astype('float32')

print(X.shape)
print(y.shape)


# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# Save the encoders and model
with open('ordinal_encoder.pkl', 'wb') as f:
    pickle.dump(ordinal_encoder, f)

with open('model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns, f)  # Save training feature names

with open('salary_model.pkl', 'wb') as f:
    pickle.dump(regressor, f)


#print(np.__version__)
#print(pd.__version__)
#print(sklearn.__version__)

