# Tele-predict
# Telecom customer churn prediction 

Overview
The Telecom Customer Churn Prediction system is a machine learning-based solution to predict whether a telecom customer will leave (churn) or stay. The system uses ensemble learning techniques such as Random Forest and XGBoost to build a predictive model based on historical customer data.

The web application is developed using Flask for the frontend, which allows users to interact with the model, input customer data, and receive churn predictions in real-time.

# Features
Predict Customer Churn: Input customer information such as account details, usage, service type, etc., and predict whether the customer will churn or not.

Random Forest & XGBoost Models: The prediction is made using ensemble learning techniques, leveraging the power of Random Forest and XGBoost algorithms for high accuracy.

Flask Web Interface: User-friendly web interface for inputting customer data and viewing predictions.

Model Evaluation: Includes model evaluation metrics such as accuracy, precision, recall, and F1-score.

# Technology Stack

# Backend:
Python (for machine learning model and Flask backend)

Flask (for serving the web application)

Scikit-learn (for machine learning models, including Random Forest)

XGBoost (for boosting algorithm)

Pandas and NumPy (for data manipulation and processing)

Matplotlib (for visualizations)

# Frontend:
Flask framework 

HTML/CSS (for basic page structure and styling)

Bootstrap (for responsive web design)

# Installation
Prerequisites

Python 3.x: Ensure you have Python installed on your system.

Virtual Environment: 

To create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Step 1: Clone the repository

Clone this repository to your local machine

Navigate to the project directory

Step 2: Install dependencies

Install the required Python libraries:

```bash
pip install Flask
pip install scikit-learn
pip install xgboost
pip install pandas
pip install numpy
pip install matplotlib
```
Step 3: Train the model 

If the model has not been pre-trained, you can train it using the provided training dataset.

Run the following script to train the Random Forest and XGBoost models:

```bash

python train_model.py
```
This will train the models and save them as serialized .pkl files, which will be used for predictions in the Flask app.

Step 4: Run the Flask application
Once dependencies are installed and the model is trained, run the Flask web application:

```bash
python app.py
```
The Flask server will start, and you can access the web interface by navigating to http://127.0.0.1:5000 in your browser.

# Usage
Open the web interface at http://127.0.0.1:5000.
Fill in the required customer information (such as partner, monthly charges, tenure, etc.).
Submit the form to receive a churn prediction result (either "Churn" or "No Churn").
View the prediction results and decision probabilities.

# Model Evaluation
Both models are evaluated using metrics like accuracy, precision, recall, and F1-score. The models are tested on a validation dataset to ensure good generalization performance.

# Example Evaluation:

python
```bash
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```
# Example of how to calculate metrics
```bash
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
```
