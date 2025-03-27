import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Load the trained model, ordinal encoder and model columns
model = pickle.load(open("model.pkl", "rb"))
ordinal_encoder = pickle.load(open("ordinal_encoder.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    test_data = {
        'work_year' : int(request.form['work_year']),
        'experience_level' : request.form['experience_level'],
        'employment_type' : request.form['employment_type'],
        'job_title' : request.form['job_title'],
        'employee_residence' : request.form['employee_residence'],
        'remote_ratio' : int(request.form['remote_ratio']),
        'company_location' : request.form['company_location'],
        'company_size' : request.form['company_size']}

    df_test = pd.DataFrame([test_data])  # Create a DataFrame from the form values
    
    # Apply ordinal encoding
    ordinal_cols = ['experience_level', 'company_size', 'remote_ratio']
    df_test[ordinal_cols] = ordinal_encoder.fit_transform(df_test[ordinal_cols])

    # Apply one-hot encoding to nominal features
    onehot_cols = ['employment_type', 'job_title', 'employee_residence', 'company_location']
    df_test = pd.get_dummies(df_test, columns=onehot_cols, drop_first=True)
   
    # Ensure test data has the same feature columns as the training data
    missing_cols = set(model_columns) - set(df_test.columns)
    for col in missing_cols:
        df_test[col] = 0  # Add missing columns with 0 values

    df_test = df_test[model_columns]  # Reorder columns to match training data

    # Predict salary

    predicted_salary = model.predict(df_test)

    return render_template('index.html', prediction_text=f'Predicted Salary: ${predicted_salary[0] :.2f}')

if __name__ == '__main__':
    app.run(debug=True)
