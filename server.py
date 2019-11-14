import json
import pandas as pd
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import re 
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from flask import Flask,render_template,url_for,request
import dill as pickle
import twint
filename = 'model_v1.pk'

with open(filename ,'rb') as f:
    loaded_model = pickle.load(f)

vectorizer = CountVectorizer(stop_words = 'english',ngram_range=(1,2))
sw = set(stopwords.words('english'))
ps = PorterStemmer()




def clean(rev):    
    rev = rev.lower()
    p = re.findall(r"@\w+",rev)
    try:
        s = rev.replace(p[0],'')
        s = s.replace('rt :','')
    except:
        s = rev
    for i in range(len(s)):
        if s[i:i+5] == 'https':
            s = s[:i]

    t = word_tokenize(s)
    t_ = [token for token in t if token not in sw]

    word = " ".join(t_)
    return word

def preProcess(text):
    text = [clean(text)]
    return text

def getTweets(user):
    c = twint.Config()
    c.Username = user
    c.Store_object = True
    c.Limit = 10
    twint.run.Search(c)
    tweets = twint.output.tweets_list
    return tweets

def getPred(user):
    df = pd.read_csv(user+"1.csv")
    df = df['tweet']
    x = [preProcess(i) for i in df]
    pred = model.predict(x)
    return pred


app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST':
        message = request.json['data']
        m = getTweets(message)
        print(message)
        data = preProcess(message)

        my_prediction = int(loaded_model.predict(data))
        print(my_prediction)
        result = {}
    if my_prediction == 0:
        result['data'] = "Neutral"
    elif my_prediction == -1:
        result['data'] = "Negative"
    elif my_prediction == 1:
        result['data'] = "Positive"
    return result

if __name__ == '__main__':
	app.run(debug=True)

    