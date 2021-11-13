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
reviews_tfidf = joblib.load('model/reviews_tfidf')
train = joblib.load('model/train')

@app.route('/')
def home():
    return "Sentiment based Recommendation System Deployment!!"

@app.route('/predict')
def predict():
    # Get values from browser
    username = request.args['reviews_username']
    d = item_final_rating.loc[username].sort_values(ascending=False)[0:20]
    top_20_products = d.index.tolist()
    
    reviews_dict = {}
    for i in top_20_products:
        idx_prod = train[train['name'] == i].index.tolist()
        reviews_vector = [reviews_tfidf[i] for i in idx_prod] 
        if(i not in reviews_dict):
            reviews_dict[i] = reviews_vector
        else:
            reviews_dict[i].append(reviews_vector) 
    
    sentiment_dict = {}

    for key,value in reviews_dict.items():
        sentiment = [int(xgboost_model.predict(i)) for i in value]
        pos_sent = 100-((reduce(lambda x,y:x+y,sentiment)/len(sentiment))*100)
        sentiment_dict[key] = pos_sent
    
    top_5 = [key for key,value in sorted(sentiment_dict.items(), key = lambda x:x[1], reverse=True)[:5]]
    print(top_5)
    
    final_list=[]
    for prod in top_5:
        final_list.append(prod)
        
    
    return jsonify(final_list)
    
if __name__ == "__main__":
    # Start Application
    app.run()