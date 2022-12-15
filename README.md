# Twitter Sentiment Analysis

## Aim :
* This project aims to perform sentiment analysis on tweets. Sentiment analysis, also known as opinion mining, is the process of determining the emotional tone behind a piece of text. In this case, we will be using tweets from the popular social media platform, Twitter.
## Getting Started:
* These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites:
* In order to run this project, you will need to have the following installed on your local machine
* nltk
* pandas
* numpy
* seaborn
* matplotlib
* tweepy to acccess twitter api
* Regular Expression
* wordcloud
* warnings
* pickle
* tensorflow
* joblib
## Technologies Used:
* Flask Framework
* HTML
* Python
* Jupyter Notebook
* Google Collab

# Process Followed:

* To extract Twitter data using the tweepy API, you can use the tweepy.Cursor class to search for tweets based on certain keywords or hashtags. For example: 
import tweepy

### Replace these with your own API keys
consumer_key = 'your-consumer-key'
consumer_secret = 'your-consumer-secret'
access_token = 'your-access-token'
access_token_secret = 'your-access-token-secret'

### Authenticate with the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

### Search for tweets containing a certain keyword or hashtag
tweets = tweepy.Cursor(api.search, q='#yourkeyword').items()

* Once you have the tweets, you can perform data processing on them, such as removing punctuation, stop words, and other irrelevant information. You can also perform data labeling by assigning a polarity (positive, negative, or neutral) to each tweet.

* To train a model for sentiment analysis, you can use a machine learning library such as Tensorflow or scikit-learn. There are several algorithms that you can use for this task, such as logistic regression, support vector classifier, and naive bayes classifier. You can try using different feature extraction techniques, such as TF-IDF or count vectorizers, to see which one performs the best for your dataset.

* Once you have trained your model, you can save it to a file using the pickle module in Python. This will allow you to use the trained model in a Flask application for live Twitter sentiment analysis.

* Here is an example of how you might use a pickled model in a Flask application:
import pickle
from flask import Flask, request

app = Flask(__name__)

# Load the pickled model
model = pickle.load(open('your-model-file.pkl', 'rb'))

@app.route('/sentiment', methods=['POST'])
def sentiment():
    # Get the tweet text from the request
    tweet = request.form['tweet']

    # Use the model to predict the sentiment of the tweet
    prediction = model.predict(tweet)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()
* With this Flask application, you can now use a trained sentiment analysis model to predict the sentiment of tweets in real-time.

### Techniques Used:
* The bag-of-words model is a way to represent text data where each word in a document is represented by a number indicating how many times it appears. This allows text to be converted into fixed-length vectors for use in machine learning models.

* Term Frequency Inverse Document Frequency (TF-IDF) is a method for extracting features from text data. It works by increasing the weight of words that appear frequently in a document but are not common across all documents in the corpus. This allows important words to be identified and used in the model.

* To train a machine learning model for sentiment analysis on text data, the data must first be labeled with its sentiment (positive, negative, or neutral). This can be done using a polarity score, which assigns a numerical value to each tweet based on its sentiment.

* Once the data is labeled, it can be used to train a model using various algorithms. In this case, the model was trained using Tensorflow, logistic regression, support vector classifier, and naive bayes classifier. These algorithms were applied on top of both count vectorizers and TF-IDF to compare their performance. The support vector classifier with TF-IDF performed the best.

## The training of the machine learning model and saving done in https://colab.research.google.com/drive/1NQ0Q-W3_A3TQ1PBzsGn4F8ytpOwa6VfO?usp=sharing

















