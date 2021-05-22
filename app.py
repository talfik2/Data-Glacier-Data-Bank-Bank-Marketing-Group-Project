import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import joblib

app = Flask(__name__, template_folder = 'template')#main klasörüm bu
model = joblib.load('model.pkl')
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')#İlk göreceğimiz şey index.html olsun

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    '''
    For rendering results on HTML
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)

    def final_countdown(prediction):
        if final_features == 0:
            return ("to buy")
        else:
            return ("not to buy")
    deneme1 = final_countdown(prediction)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text = 'For the given values, this customer is more likely  {} the desired product of the bank'.format(deneme1))

if __name__ == "__main__":
    app.run(port = 12345, debug = True)
