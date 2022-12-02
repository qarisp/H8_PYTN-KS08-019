from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Logistic Regression Pickle
model1 = pickle.load(open("model/lreg_model.pkl", "rb"))
# Scaler initialization
scaler = StandardScaler()

app = Flask(__name__, template_folder="templates")


@app.route("/")
def main():
    return render_template('index.html')


@app.route('/predict_1', methods=['POST'])
def predict_1():
    '''
    For rendering results on HTML GUI
    '''
    Rainfall = float(request.form["Rainfall"])
    WindGustDir = int(request.form["WindGustDir"])
    WindGustSpeed = float(request.form["WindGustSpeed"])
    Humidity3pm = float(request.form["Humidity3pm"])
    Pressure9am = float(request.form["Pressure9am"])
    Cloud3pm = float(request.form["Cloud3pm"])
    Temp9am = float(request.form["Temp9am"])
    Temp3pm = float(request.form["Temp3pm"])
    Year = int(request.form["Year"])
    Month = int(request.form["Month"])
    Day = int(request.form["Day"])

    data_list = [[
        Rainfall,
        WindGustDir,
        WindGustSpeed,
        Humidity3pm,
        Pressure9am,
        Cloud3pm,
        Temp9am,
        Temp3pm,
        Year,
        Month,
        Day
    ]]
    pred_scaller = scaler.fit_transform(data_list)
    prediction = model1.predict(pred_scaller)

    output = {
        0: "Tidak Hujan",
        1: "Hujan"
    }

    return render_template('index.html', prediction_text='Prediksi Cuaca Hari Besok adalah : {}'.format(output[prediction[0]]))


if __name__ == '__main__':
    app.run(debug=True)
