# deploy-heroku

Heroku deployed app link - https://deploy-recoystem-heroku.herokuapp.com/

Gtihub link for all the code - https://github.com/bkkilambi/deploy-heroku

Files Description: 

Project_final.ipynb : All the model training, finding best models, code for recommendation system etc are in the Project_final.ipynb file. 

app.py : Contains all the code to get username from the user from UI and get the best sentiment based recommendations for that user. 

model folder : Contains the xgboost ML model for predicting semtiment, user_ratings for extracting recommendations of products for the user and a TFIDF vector of all reviews 
from the train dataset which are then given to the ML model to get sentiments (for the top 20 recommended products) and then the products which have the highest 
percentage of positive sentiments are chosen. 

requirements.txt : All the required/dependent packages to be installed
