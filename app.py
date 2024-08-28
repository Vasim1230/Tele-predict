from flask import Flask, render_template, jsonify,request, session,redirect, url_for,flash
import numpy as np
import pandas as pd
import pickle
from flask_sqlalchemy import SQLAlchemy

from werkzeug.security import generate_password_hash, check_password_hash

 
model = pickle.load(open('telecom.pkl', 'rb')) 

app = Flask(__name__)
import os

app.secret_key = b'\xc1\x96 \xc3\xf5\xdd\xf0\x9e\xd90A\xcd\xdb\xf0\xdcu\xe7F\x994x\r\x9c\xed'
# Update the below SQLALCHEMY_DATABASE_URI with your database details
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
with app.app_context():
    db.create_all()

@app.route('/',methods=['GET','POST'])
def design():
    
    return render_template('homepage_1.html')    


@app.route('/login', methods=['GET','POST'])
def login(): 
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            flash('Login successful!', 'success')
            return render_template('predictionpage_3.html')
        else:
            flash('Invalid username or password', 'error')
            return render_template('loginpage_2.html')   
    
        
    

    

def ang():
    render_template('ang.png')



@app.route('/predict', methods=['GET', 'POST'])
def man():
    return render_template('predictionpage_3.html')
 
 
@app.route('/result', methods=['GET', 'POST'])
def home():
    data1 = request.form['SeniorCitizen']
    data2 = request.form['Partner']
    data3 = request.form['Dependents']
    data4 = request.form['tenure']
    data5 = request.form['OnlineSecurity']
    data6 = request.form['OnlineBackup']
    data7 = request.form['DeviceProtection']
    data8 = request.form['TechSupport']
    data9 = request.form['Contract']
    data10 = request.form['PaperlessBilling']
    data11 = request.form['MonthlyCharges']
    data12 = request.form['TotalCharges']
    
   
    
   
 
    arr = np.array([[data1, data2, data3, data4, data5, data6,data7,data8,data9,data10,data11,data12  ]])
    prediction = model.predict(arr)
    result = "A Churn" if prediction == 1 else "Not a Churn"
    

    return render_template('resultpage_4.html', result=result, data=prediction, data1=data1, data2=data2, data3=data3,
                           data4=data4, data5=data5, data6=data6,data7=data7, data8=data8,data9=data9,data10=data10,data11=data11,data12=data12)

if __name__ == "__main__":
    app.run(debug=True)