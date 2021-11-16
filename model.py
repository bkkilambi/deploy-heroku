import joblib
import numpy as np
from flask import app, render_template
import pandas as pd
from functools import reduce

def make_recos(username):
    # Load the model from the file
    xgboost_model = joblib.load('model/xgboost_model.pkl')
    item_final_rating = joblib.load('model/item_final_rating')
    prod_tfidf = joblib.load('model/prod_tfidf')

    d = item_final_rating.loc[username].sort_values(ascending=False)[0:20]
    top_20_products = d.index.tolist()
    sentiment_dict={}
    for i in top_20_products:
        reviews_list = prod_tfidf[i]
        sentiment = [xgboost_model.predict(rev) for rev in reviews_list]
        pos_sent = 100-((reduce(lambda x,y:x+y,sentiment)/len(sentiment))*100)
        sentiment_dict[i] = pos_sent
    
    top_5 = [key for key,value in sorted(sentiment_dict.items(), key = lambda x:x[1], reverse=True)[:5]]

    return top_5


