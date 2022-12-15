import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC


from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

import tweepy #to access the twitter api
import pandas as pd #for basic data operations

# Importing the keys from twitter api

# Search for the Term and define number of tweets 
# searchTerm = input("Enter Keyword/Tag to search about: ")
consumerKey = "6Xx7Z3LnGcVR2yT246vh0FX54"
consumerSecret = "NPYutG3hpLSTAxSpCBbJ17Ysu8KKZXGvTQqIj0fPlhUlnId9vX"
accessToken = "2578006729-I42EMRgLnmuDs1cPCcLfJeS3EQ3guyjhFVuR4af"
accessTokenSecret = "0ozPKKjVOfCpY4beXXMomFYz2npQimnbcyhOkfA5bm5bK"

# Establish the connection with twitter API
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)
tweets_list = [] 


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    for tweet in api.search_tweets(q='life', count=1):
            print(tweet.text)
            # Adding to list that contains all tweets
            tweets_list.append({
                                'tweet_text': tweet.text})
    df = pd.DataFrame.from_dict(tweets_list)

    print(df['tweet_text'].values)
    vect_model=pickle.load(open('vector.pkl','rb')) 
    pickled_model=pickle.load(open('SVCmodel.pkl','rb')) 

    tweettext = df['tweet_text'].values
    prediction = pickled_model.predict(vect_model.transform(tweettext))
    return render_template("home.html",prediction_text="The prediction for the tweet {}".format(tweettext[0]),prediction ="{}".format(prediction[0]) )



if __name__=="__main__":
    app.run(debug=True)
   
     