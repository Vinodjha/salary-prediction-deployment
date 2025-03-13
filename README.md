# Salary Prediction Model

## Overview
This project demonstrates a simple machine learning model that predicts employee salary based on experience, test scores, and interview scores. The model is trained using the **Linear Regression** algorithm and is deployed using **Flask** as a web application. The trained model is saved using **Pickle** for reuse.

## Project Structure
```
│── model.py       # Script to generate sample data, train the model, and save it using Pickle
│── app.py         # Flask web application to serve predictions
│── model.pkl      # Saved machine learning model
│── templates/
│   ├── index.html # Frontend HTML file for user input and displaying predictions
│── static/        # (Optional) Contains CSS, JS, or images for styling
│── requirements.txt # Dependencies list
│── README.md      # Project documentation
```

## Features
- Generates random training data for salary prediction.
- Trains a **Linear Regression** model using **scikit-learn**.
- Saves the trained model using **Pickle** for deployment.
- A **Flask-based** web interface to accept user input and predict salary.
- Easily deployable on local machines or cloud platforms like **Render** or **Heroku**.

## Installation

### Prerequisites
Ensure you have **Python 3.x** installed along with the necessary libraries.

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model training script to generate `model.pkl`:
   ```bash
   python model.py
   ```
4. Start the Flask application:
   ```bash
   python app.py
   ```
5. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage
1. Enter values for **experience**, **test score**, and **interview score** in the web UI.
2. Click the **Predict** button.
3. The estimated salary is displayed on the screen.

## Dependencies
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `Flask`
- `pickle`

## Deployment
To deploy on a cloud platform:
1. Use a service like **Render** or **Heroku**.
2. Configure dependencies using `requirements.txt`.
3. Ensure `model.pkl` is included in the project directory.

## License
This project is open-source and free to use for educational and research purposes.

---
For any issues or contributions, feel free to submit a pull request or raise an issue!
