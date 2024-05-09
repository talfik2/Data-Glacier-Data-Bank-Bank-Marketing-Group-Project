import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import joblib

app = Flask(__name__, template_folder = 'template')
# template_folder = 'template' -> look for html templates in the dir called 'template' 
model = joblib.load('model.pkl')
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/') # when someone goes to the main site, below function(home) will run.  
# Since function home will return index.html, when someone goes to the main site, they'll see the index.html
# Nothing but / because below function will call the main page
def home():
    return render_template('index.html')#İlk göreceğimiz şey index.html olsun

@app.route('/predict',methods=['POST', 'GET']) 
# run the below function(predict) when the form is submitted(when someone clicked the button)
# '/predict' execute <domain>/predict
# www.facebook.com = root, www.facebook.com/login = Log in part of the root
def predict():
    '''
    For rendering results on HTML
    '''
    int_features = [int(x) for x in request.form.values()]
    # request.form.values() = Retrieve the inputs and convert them to intigers for ML model to work on 
    final_features = [np.array(int_features)] # convert the input values to Np arrays

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
