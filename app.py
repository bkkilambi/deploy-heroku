# import Flask class from the flask module
from flask import Flask, request
import joblib
import numpy as np
from flask import app, render_template
import pandas as pd
from functools import reduce
from flask import jsonify


# Create Flask object to run
app = Flask(__name__)

# Load the model from the file
xgboost_model = joblib.load('model/xgboost_model.pkl')
item_final_rating = joblib.load('model/item_final_rating')
prod_tfidf = joblib.load('model/prod_tfidf')

@app.route('/')
def home():
    #return "Sentiment based Recommendation System Deployment!!"
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Get values from browser
    if (request.method == 'POST'):
        #username = request.args['reviews_username']
        username = request.get_data().decode('utf-8')
        if username is not None:
            username = username.split('=')[1]
        print(username)
        d = item_final_rating.loc[username].sort_values(ascending=False)[0:20]
        top_20_products = d.index.tolist()
        #print(top_20_products)

        sentiment_dict={}
        for i in top_20_products:
            reviews_list = prod_tfidf[i]
            sentiment = [xgboost_model.predict(rev) for rev in reviews_list]
            pos_sent = 100-((reduce(lambda x,y:x+y,sentiment)/len(sentiment))*100)
            sentiment_dict[i] = pos_sent
        #print(sentiment_dict)

        top_5 = [key for key,value in sorted(sentiment_dict.items(), key = lambda x:x[1], reverse=True)[:5]]

        final_list=[]

        for prod in top_5:
            final_list.append(prod)
        print(final_list)

        return render_template('index.html', prediction_text='Recommendations : {}'.format(final_list))
    else:
        return render_template('index.html')

    #return jsonify(final_list)


if __name__ == "__main__":
    # Start Application
    app.run()