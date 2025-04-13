from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime

app = Flask(__name__)
model = load_model('heart_disease_mode1l.h5')

@app.route('/')
def home():
    year = datetime.now().year
    return render_template('index.html', year=year)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        BMI = float(request.form['BMI'])
        Fruits = int(request.form['Fruits'])
        GenHlth = int(request.form['GenHlth'])
        MentHlth = int(request.form['MentHlth'])
        PhysHlth = int(request.form['PhysHlth'])
        Age = int(request.form['Age'])
        Education = int(request.form['Education'])
        Income = int(request.form['Income'])

        features = np.array([[BMI, Fruits, GenHlth, MentHlth, PhysHlth, Age, Education, Income]])
        prediction = model.predict(features)[0][0]

        if prediction >= 0.5:
            message = 'Result: High Risk of Heart Disease'
            status = 'high-risk'
        else:
            message = 'Result: Low Risk of Heart Disease'
            status = 'low-risk'

        return jsonify({'message': message, 'status': status})

    except Exception as e:
        return jsonify({'message': 'Error: ' + str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True)
