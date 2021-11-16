# import Flask class from the flask module
from flask import Flask, request
import joblib
import numpy as np
from flask import app, render_template
import pandas as pd
from functools import reduce
from flask import jsonify
from model import *

# Create Flask object to run
app = Flask(__name__)

@app.route('/')
def home():
    #return "Sentiment based Recommendation System Deployment!!"
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Get values from browser
    if (request.method == 'POST'):
        username = request.get_data().decode('utf-8')
        if username is not None:
            username = username.split('=')[1]
        print(username)

        top_5 = make_recos(username)
        final_list=[]

        for prod in top_5:
            final_list.append(prod)
        print(final_list)

        return render_template('index.html', prediction_text='Recommendations : {}'.format(final_list))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    # Start Application
    app.run()
